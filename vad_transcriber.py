"""
vad_transcriber.py
Unified worker-pool Whisper architecture:
  • N identical workers (configured via config.json NUM_WHISPER_PROCESSES)
  • Single asyncio.Queue — scan loop pushes ready chunks, workers pull and transcribe
"""

import asyncio
import logging
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np

from audio_recorder import AudioBufferManager

logger = logging.getLogger(__name__)

SAMPLE_RATE              = 16000
FLUSH_INTERVAL           = 0.3
MIN_AUDIO_SECS           = 1.5   # minimum audio before considering transcription
PRE_ROLL_SECS            = 0.8
MIN_ENERGY               = 0.002
SILENCE_ADVANCE_AFTER    = 6    # consecutive silent ticks before advancing cursor (~1.8s)
TAIL_SILENCE_SECS        = 1.2  # tail silence in pending audio → phrase boundary
MAX_WAIT_SECS            = 5.0  # force transcription if no phrase boundary for this long
MAX_AUDIO_SECS           = 30.0 # cap chunk size sent to Whisper to avoid worker saturation
WHISPER_TIMEOUT_SECS     = 25   # kill a hung Whisper call after this many seconds
LIVE_RMS_DECAY           = 0.993 # per 30 ms chunk → half-life ~3 s


_STOPWORDS = {
    # Spanish articles, prepositions, conjunctions, pronouns
    "la", "el", "los", "las", "de", "del", "en", "a", "al", "y", "o", "e",
    "que", "se", "es", "son", "no", "ni", "si", "su", "sus", "un", "una",
    "lo", "le", "les", "me", "mi", "te", "tu", "yo", "él", "ella",
    # English equivalents
    "the", "a", "an", "is", "are", "of", "in", "on", "and", "or", "not",
    "to", "it", "he", "she", "we", "i", "you", "my", "your", "its",
}

def _looks_like_hallucination(text: str) -> bool:
    """Catch repetitive / counting hallucinations that Whisper produces on near-silent audio."""
    # ── Pattern 1: dot-chained repetition without spaces ("I...I...I..." or "You.You.You.") ──
    punct_tokens = [t.strip() for t in re.split(r'[.\s]+', text) if t.strip()]
    if len(punct_tokens) >= 6:
        counts_p = Counter(t.lower() for t in punct_tokens)
        top_word, top_p = counts_p.most_common(1)[0]
        if top_word not in _STOPWORDS and top_p / len(punct_tokens) > 0.5:
            return True

    # ── Pattern 2: word-level repetition ──
    words = text.split()
    if len(words) < 8:          # short phrases repeat naturally in Spanish
        return False
    tokens = [re.sub(r"[.,!?;:\-¿¡]", "", w).lower() for w in words]
    counts = Counter(tokens)
    top_word, top_count = counts.most_common(1)[0]
    # Require at least 3 occurrences AND the word is not a stopword
    if top_count >= 3 and top_word not in _STOPWORDS and top_count / len(tokens) > 0.40:
        return True

    # ── Pattern 3: clause-level repetition (same clause 2+ times after commas/semicolons) ──
    clauses = [re.sub(r"[.,!?;:\-¿¡]", "", c).strip().lower()
               for c in re.split(r"[,;]", text) if c.strip()]
    if len(clauses) >= 3:
        top_clause, top_n = Counter(clauses).most_common(1)[0]
        if top_n >= 2 and len(top_clause.split()) >= 3:
            return True

    # ── Pattern 4: 3-gram repetition across the full text ──
    if len(tokens) >= 9:
        trigrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        top_tri, top_tri_n = Counter(trigrams).most_common(1)[0]
        if top_tri_n >= 3:
            return True

    return False


@dataclass
class TranscriptSegment:
    device_id:    int
    speaker_name: str
    text:         str
    language:     str
    confidence:   float
    timestamp:    float = field(default_factory=time.time)
    is_partial:   bool  = False
    chunk_id:     Optional[str] = None
    rms_volume:        float = 0.0
    dominant_peer_rms: float = 0.0   # max live_rms of all other devices at queue time


class WhisperTranscriber:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="int8"):
        self.model_size   = model_size
        self.device       = device
        self.compute_type = compute_type
        self.model        = None
        self._stub_mode   = False
        self._busy        = False
        self._executor    = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self.last_detected_language:             Optional[str] = None
        self.last_detected_language_probability: float         = 0.0

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self._executor, self._load_sync)
            logger.info(f"Whisper '{self.model_size}' loaded on {self.device}")
        except ImportError:
            logger.warning("faster-whisper not installed — stub mode")
            self._stub_mode = True
        except Exception as e:
            logger.error(f"Whisper load failed: {e} — stub mode")
            self._stub_mode = True

    def _load_sync(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=1,
            download_root="./models",
        )

    async def transcribe(self, audio: np.ndarray) -> Optional[TranscriptSegment]:
        if self._stub_mode:
            return self._stub(audio)
        if self.model is None or self._busy:
            return None
        self._busy = True
        try:
            loop = asyncio.get_event_loop()
            fut  = loop.run_in_executor(self._executor, lambda: self._transcribe_sync(audio))
            return await asyncio.wait_for(fut, timeout=WHISPER_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            # The GPU thread is stuck — replace the executor so the pool is usable again.
            # The old stuck thread will eventually die on its own.
            logger.warning(f"Whisper timeout after {WHISPER_TIMEOUT_SECS}s — recycling executor")
            self._executor.shutdown(wait=False)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
            return None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
        finally:
            self._busy = False

    def _transcribe_sync(self, audio: np.ndarray) -> Optional[TranscriptSegment]:
        segments, info = self.model.transcribe(
            audio,
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            temperature=0.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=" ",
            max_new_tokens=200,   # cap token generation — prevents infinite loops on bad audio
        )

        self.last_detected_language             = info.language
        self.last_detected_language_probability = info.language_probability

        if info.language not in ("en", "es"):
            return None
        if info.language_probability < 0.25:
            return None

        texts, total_logprob, count = [], 0.0, 0
        rejected = []
        for seg in segments:
            t              = seg.text.strip()
            avg_logprob    = getattr(seg, "avg_logprob", -1.0)
            no_speech_prob = getattr(seg, "no_speech_prob", 1.0)
            if t and avg_logprob >= -1.1 and no_speech_prob < 0.6:
                texts.append(t)
                total_logprob += avg_logprob
                count         += 1
            elif t:
                rejected.append(f"logp={avg_logprob:.2f} nsp={no_speech_prob:.2f} '{t[:30]}'")

        if not texts:
            if rejected:
                logger.info(f"All segs filtered: {rejected[:3]}")
            return None

        HALLUCINATIONS = {
            "thank you.", "thanks for watching.", "thanks for watching",
            "thank you for watching.", "thank you for watching",
            "okay.", "ok.", "bye.", "bye bye.", "gracias.", "¡gracias!", "gracias!",
            "adiós.", "adios.", "de nada.", "sí.", "no.", "bien.", "claro.", "bueno.",
            "you.", "thank you so much.", "thanks.", "welcome.", "we'll be right back.",
            "subscribete.", "suscríbete.", "like and subscribe.", "see you next time.",
            "so, bye.", "all right.", "thumbs out.", "alright.", "so bye.",
            "all right!", "so, bye!", "thumbs out!",
            "esta es una conversación en inglés.",
            "esta es una conversación en inglés",
            "esta es una conversación en español e inglés.",
            "esta es una conversación en español e inglés",
            "esta es una conversación en español y inglés.",
            "esta es una conversación en español y inglés",
            "this is a conversation in english and spanish.",
            "this is a conversation in english and spanish",
            "gracias por ver el video.", "gracias por ver el video",
            "gracias por ver.", "gracias por ver",
            "¡gracias por ver el video!", "¡gracias por ver!",
            "no olvides suscribirte.", "no olvides suscribirte",
            "suscríbete al canal.", "suscríbete al canal",
            "hasta la próxima.", "hasta la próxima",
            "hasta pronto.", "hasta pronto",
            "let's pray.", "let's pray", "let us pray.", "let us pray",
            "amen.", "amen", "amén.", "amén",
            "gracias a dios.", "gracias a dios",
            "dios te bendiga.", "dios te bendiga",
            "en el nombre de dios.", "en el nombre de dios",
            "en el nombre del padre.", "en el nombre del padre",
            "en el nombre del padre, del hijo y del espíritu santo.",
            "en el nombre del padre, del hijo y del espíritu santo",
            "padre nuestro.", "padre nuestro",
            "la iglesia.", "la iglesia", "en la iglesia.", "en la iglesia",
            "señor.", "señores.", "el señor.",
            "gloria a dios.", "gloria a dios",
            "aleluya.", "aleluya", "hallelujah.", "hallelujah",
            "dios mío.", "dios mío", "ay, dios mío.", "ay, dios mío",
            "bendito sea dios.", "bendito sea dios",
            # French hallucinations (Whisper misclassifies French audio as es/en)
            "oui.", "oui", "oui oui.", "oui, oui.", "oui oui", "oui, oui",
            "merci.", "merci", "bonjour.", "bonjour", "bonsoir.", "bonsoir",
            "au revoir.", "au revoir", "s'il vous plaît.", "s'il vous plaît",
        }
        full_text = " ".join(texts).strip()
        if full_text.lower() in HALLUCINATIONS:
            return None
        if _looks_like_hallucination(full_text):
            logger.info(f"Discarded repetitive hallucination: {repr(full_text[:80])}")
            return None

        return TranscriptSegment(
            device_id=-1, speaker_name="",
            text=full_text,
            language=info.language,
            confidence=float(np.exp(total_logprob / count)) if count else 0.5,
        )

    def _stub(self, audio: np.ndarray) -> Optional[TranscriptSegment]:
        import random
        if len(audio) / SAMPLE_RATE < 0.3:
            return None
        samples = [("Hola, como estas?", "es"), ("Can you hear me?", "en")]
        text, lang = random.choice(samples)
        return TranscriptSegment(
            device_id=-1, speaker_name="", text=text, language=lang, confidence=0.91,
        )


class VADTranscriptionPipeline:
    def __init__(self, model_size="large-v3", cuda_device="cuda", compute_type="int8"):
        self.model_size   = model_size
        self.cuda_device  = cuda_device
        self.compute_type = compute_type

        # Load worker count from config.json
        import json as _json, os as _os
        _cfg_path = _os.path.join(_os.path.dirname(__file__), "config.json")
        try:
            with open(_cfg_path) as _f:
                _cfg = _json.load(_f)
        except Exception:
            _cfg = {}
        self._num_workers = int(_cfg.get("NUM_WHISPER_PROCESSES", 5))

        self._workers: List[WhisperTranscriber] = [
            WhisperTranscriber(model_size, cuda_device, compute_type)
            for _ in range(self._num_workers)
        ]
        self._work_queue:     asyncio.Queue          = asyncio.Queue()
        self._queued_devices: set                    = set()
        self._worker_tasks:   List[asyncio.Task]     = []
        self._scan_task:      Optional[asyncio.Task] = None

        # Per-device state
        self._speaker_names:  Dict[int, str]   = {}
        self._device_active:  Dict[int, bool]  = {}
        self._silence_streak: Dict[int, int]   = {}
        self._noise_gate:     Dict[int, float] = {}
        self._last_tx_time:   Dict[int, float] = {}
        self._live_rms:       Dict[int, float] = {}

        self._instant_mode = False
        self._fast_worker: Optional[WhisperTranscriber] = None

        self.buffer_mgr:              Optional[AudioBufferManager] = None
        self.on_transcript:           Optional[Callable]           = None
        self.on_transcript_partial:   Optional[Callable]           = None
        self.on_transcript_remove:    Optional[Callable]           = None
        self.on_device_inactive:      Optional[Callable]           = None
        self.on_device_active:        Optional[Callable]           = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def set_instant_mode(self, enabled: bool) -> None:
        self._instant_mode = enabled
        logger.info(f"Transcription mode: {'instant' if enabled else 'precise'}")

    async def load(self) -> None:
        logger.info(f"Loading {self._num_workers} Whisper workers…")
        for i, w in enumerate(self._workers):
            logger.info(f"  Loading worker {i}…")
            await w.load()
        logger.info("All Whisper workers ready")
        logger.info("Loading fast Whisper worker (small/cpu) for Instant mode…")
        self._fast_worker = WhisperTranscriber("small", "cpu", "int8")
        await self._fast_worker.load()
        logger.info("Fast worker ready")
        self._scan_task = asyncio.create_task(self._buffer_scan_loop(), name="buf_scan")
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(i), name=f"worker_{i}")
            for i in range(self._num_workers)
        ]

    async def shutdown(self) -> None:
        if self._scan_task:
            self._scan_task.cancel()
        for t in self._worker_tasks:
            t.cancel()
        for w in self._workers:
            w.shutdown()
        if self._fast_worker:
            self._fast_worker.shutdown()

    # ── Speaker registration ───────────────────────────────────────────────

    def register_speaker(self, device_id: int, name: str) -> None:
        self._speaker_names[device_id]  = name
        self._device_active[device_id]  = True
        self._silence_streak[device_id] = 0
        self._noise_gate[device_id]     = MIN_ENERGY
        self._last_tx_time[device_id]   = time.time()
        logger.info(f"Registered: {name} (device {device_id})")

    def unregister_speaker(self, device_id: int) -> None:
        self._queued_devices.discard(device_id)
        for d in (self._speaker_names, self._device_active, self._silence_streak,
                  self._noise_gate, self._last_tx_time, self._live_rms):
            d.pop(device_id, None)

    def update_speaker_name(self, device_id: int, name: str) -> None:
        self._speaker_names[device_id] = name

    def set_noise_gate(self, device_id: int, value: float) -> None:
        self._noise_gate[device_id] = max(0.001, min(0.5, value))
        logger.info(f"Device {device_id} noise gate → {value:.4f}")

    # ── Audio ingestion (no-op: buffer managed by app.py) ────────────────

    async def process_audio_chunk(self, device_id: int, pcm: np.ndarray) -> None:
        chunk_rms = float(np.sqrt(np.mean(pcm ** 2)))
        self._live_rms[device_id] = max(chunk_rms, self._live_rms.get(device_id, 0.0) * LIVE_RMS_DECAY)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _handle_silence(self, device_id: int, audio: np.ndarray, buf) -> bool:
        """Returns True only when the entire buffer (body + tail) is silent.

        Splitting body/tail prevents short words followed by silence (e.g. "bella"
        after a pause) from being classified as silence and discarded.
        """
        threshold = self._noise_gate.get(device_id, MIN_ENERGY)
        tail_n = int(TAIL_SILENCE_SECS * SAMPLE_RATE)
        if len(audio) > tail_n:
            body_rms = float(np.sqrt(np.mean(audio[:-tail_n] ** 2)))
            tail_rms = float(np.sqrt(np.mean(audio[-tail_n:] ** 2)))
        else:
            body_rms = tail_rms = float(np.sqrt(np.mean(audio ** 2)))

        if body_rms >= threshold or tail_rms >= threshold:
            self._silence_streak[device_id] = 0
            if not self._device_active.get(device_id, True):
                self._device_active[device_id] = True
                if self.on_device_active:
                    asyncio.ensure_future(self.on_device_active(device_id))
            return False

        streak = self._silence_streak.get(device_id, 0) + 1
        self._silence_streak[device_id] = streak
        if streak >= 60 and self._device_active.get(device_id, True):
            self._device_active[device_id] = False
            if self.on_device_inactive:
                asyncio.ensure_future(self.on_device_inactive(device_id))
        if streak >= SILENCE_ADVANCE_AFTER:
            pre_roll = int(PRE_ROLL_SECS * SAMPLE_RATE)
            advance = max(0, len(audio) - pre_roll)
            buf.mark_transcribed(advance)
            logger.debug(
                f"dev={device_id} silence streak={streak} "
                f"body_rms={body_rms:.4f} tail_rms={tail_rms:.4f} "
                f"pending={len(audio)/SAMPLE_RATE:.2f}s advance={advance/SAMPLE_RATE:.2f}s"
            )
        return True

    # ── Scan loop + worker loops ──────────────────────────────────────────

    async def _buffer_scan_loop(self) -> None:
        """Scan all registered devices every FLUSH_INTERVAL; push ready chunks to queue."""
        while True:
            await asyncio.sleep(FLUSH_INTERVAL)
            if not self.buffer_mgr:
                continue
            for device_id in list(self._speaker_names.keys()):
                if device_id in self._queued_devices:
                    continue
                buf = self.buffer_mgr.get_buffer(device_id)
                if buf is None:
                    continue
                audio = buf.get_pending()
                if audio is None or len(audio) < int(MIN_AUDIO_SECS * SAMPLE_RATE):
                    continue
                if self._handle_silence(device_id, audio, buf):
                    continue
                tail = audio[-int(TAIL_SILENCE_SECS * SAMPLE_RATE):]
                tail_rms = float(np.sqrt(np.mean(tail ** 2)))
                threshold = self._noise_gate.get(device_id, MIN_ENERGY)
                at_boundary = tail_rms < threshold
                force = (time.time() - self._last_tx_time.get(device_id, 0)) > MAX_WAIT_SECS
                if not at_boundary and not force:
                    continue
                n_full = len(audio)
                max_samples = int(MAX_AUDIO_SECS * SAMPLE_RATE)
                if n_full > max_samples:
                    audio = audio[-max_samples:]  # keep most recent audio (contains speech)
                n_samples = len(audio)
                dominant_peer_rms = max(
                    (v for k, v in self._live_rms.items() if k != device_id),
                    default=0.0,
                )
                buf.mark_transcribed(n_full)   # advance cursor past ALL pending (including discarded prefix)
                self._queued_devices.add(device_id)
                await self._work_queue.put((
                    device_id, audio, n_samples,
                    self._speaker_names.get(device_id, f"Speaker {device_id}"),
                    dominant_peer_rms,
                ))

    async def _worker_loop(self, worker_idx: int) -> None:
        """Pull work items from the queue and transcribe them."""
        whisper = self._workers[worker_idx]
        while True:
            device_id, audio, n_samples, speaker_name, dominant_peer_rms = \
                await self._work_queue.get()
            try:
                await self._transcribe_and_emit(
                    audio, n_samples, device_id, speaker_name, worker_idx,
                    dominant_peer_rms,
                )
            finally:
                self._queued_devices.discard(device_id)
                self._work_queue.task_done()

    async def _transcribe_and_emit(self, audio, n_samples, device_id, speaker_name, worker_idx,
                                    dominant_peer_rms: float = 0.0):
        whisper    = self._workers[worker_idx]
        rms_volume = float(np.sqrt(np.mean(audio[:n_samples] ** 2)))
        logger.info(f"dev={device_id} → Whisper worker={worker_idx} audio={n_samples/SAMPLE_RATE:.2f}s rms={rms_volume:.4f} peer_rms={dominant_peer_rms:.4f}")

        chunk_id       = None
        partial_emitted = False

        if self._instant_mode and self._fast_worker:
            chunk_id = uuid.uuid4().hex[:12]
            fast_seg = await self._fast_worker.transcribe(audio)
            self._last_tx_time[device_id] = time.time()
            if fast_seg and fast_seg.text.strip():
                fast_seg.device_id         = device_id
                fast_seg.speaker_name      = speaker_name
                fast_seg.rms_volume        = rms_volume
                fast_seg.dominant_peer_rms = dominant_peer_rms
                fast_seg.is_partial        = True
                fast_seg.chunk_id          = chunk_id
                if self.on_transcript_partial:
                    try:
                        await self.on_transcript_partial(fast_seg)
                        partial_emitted = True
                    except Exception as e:
                        logger.error(f"Partial transcript callback error: {e}")

        segment = await whisper.transcribe(audio)
        # mark_transcribed was already called at queue time in _buffer_scan_loop
        self._last_tx_time[device_id] = time.time()

        lang = whisper.last_detected_language
        prob = whisper.last_detected_language_probability
        text = segment.text if segment else None
        logger.info(f"dev={device_id} lang={lang}({prob:.2f}) → {repr(text)}")
        if segment and segment.text.strip():
            segment.device_id         = device_id
            segment.speaker_name      = speaker_name
            segment.rms_volume        = rms_volume
            segment.dominant_peer_rms = dominant_peer_rms
            segment.chunk_id          = chunk_id if partial_emitted else None
            if self.on_transcript:
                try:
                    await self.on_transcript(segment)
                except Exception as e:
                    logger.error(f"Transcript callback error: {e}")
        elif partial_emitted and self.on_transcript_remove:
            # Large model filtered the audio (hallucination) — clean up the partial
            try:
                await self.on_transcript_remove(chunk_id, device_id)
            except Exception as e:
                logger.error(f"Transcript remove callback error: {e}")
