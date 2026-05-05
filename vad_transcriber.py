"""
vad_transcriber.py
5-pool Whisper architecture:
  • Pools 0-3  — exclusive (one dedicated device each)
  • Pool  4    — shared   (remaining devices, round-robin queue)

Pool assignments are dynamic and controllable via API at runtime.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np

from audio_recorder import AudioBufferManager

logger = logging.getLogger(__name__)

SAMPLE_RATE              = 16000
FLUSH_INTERVAL           = 0.5
MIN_AUDIO_SECS           = 2.5
PRE_ROLL_SECS            = 0.8
MIN_ENERGY               = 0.002
SILENCE_ADVANCE_AFTER    = 6   # consecutive silent ticks before advancing cursor (~3s)

NUM_POOLS       = 5
EXCLUSIVE_SLOTS = 4   # pools 0-3
SHARED_POOL     = 4   # pool 4


@dataclass
class TranscriptSegment:
    device_id:    int
    speaker_name: str
    text:         str
    language:     str
    confidence:   float
    timestamp:    float = field(default_factory=time.time)
    is_partial:   bool  = False


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
            return await loop.run_in_executor(self._executor, lambda: self._transcribe_sync(audio))
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
        )

        self.last_detected_language             = info.language
        self.last_detected_language_probability = info.language_probability

        if info.language not in ("en", "es"):
            return None
        if info.language_probability < 0.25:
            return None

        texts, total_logprob, count = [], 0.0, 0
        for seg in segments:
            t              = seg.text.strip()
            avg_logprob    = getattr(seg, "avg_logprob", -1.0)
            no_speech_prob = getattr(seg, "no_speech_prob", 1.0)
            if t and avg_logprob > -0.8 and no_speech_prob < 0.6:
                texts.append(t)
                total_logprob += avg_logprob
                count         += 1

        if not texts:
            return None

        HALLUCINATIONS = {
            "thank you.", "thanks for watching.", "thanks for watching",
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
        }
        full_text = " ".join(texts).strip()
        if full_text.lower() in HALLUCINATIONS:
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

        # 5 fixed Whisper pools (0-3 exclusive, 4 shared)
        self._pools: List[WhisperTranscriber] = [
            WhisperTranscriber(model_size, cuda_device, compute_type)
            for _ in range(NUM_POOLS)
        ]

        # Slot → device_id (None = empty)
        self._exclusive_slots: Dict[int, Optional[int]] = {i: None for i in range(EXCLUSIVE_SLOTS)}

        # device_id → pool index (0-3 exclusive, 4 shared)
        self._device_pool: Dict[int, int] = {}

        # Flush tasks
        self._exclusive_flush_tasks: Dict[int, asyncio.Task] = {}
        self._shared_flush_task: Optional[asyncio.Task]      = None

        # Per-device state
        self._speaker_names:   Dict[int, str]           = {}
        self._device_active:   Dict[int, bool]          = {}
        self._silence_streak:  Dict[int, int]           = {}
        self._noise_gate:      Dict[int, float]         = {}
        self._device_language: Dict[int, Optional[str]] = {}
        self._language_streak: Dict[int, int]           = {}

        self.buffer_mgr:         Optional[AudioBufferManager] = None
        self.on_transcript:      Optional[Callable]           = None
        self.on_device_inactive: Optional[Callable]           = None
        self.on_device_active:   Optional[Callable]           = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def load(self) -> None:
        logger.info(f"Loading {NUM_POOLS} Whisper pools…")
        for i, pool in enumerate(self._pools):
            label = f"exclusive slot {i}" if i < EXCLUSIVE_SLOTS else "shared pool"
            logger.info(f"  Loading pool {i} ({label})…")
            await pool.load()
        logger.info("All Whisper pools ready")
        self._shared_flush_task = asyncio.create_task(
            self._shared_flush_loop(), name="flush_shared"
        )

    async def load_all_whispers(self) -> None:
        pass  # pools are loaded in load()

    async def shutdown(self) -> None:
        for task in self._exclusive_flush_tasks.values():
            task.cancel()
        if self._shared_flush_task:
            self._shared_flush_task.cancel()
        for pool in self._pools:
            pool.shutdown()

    # ── Speaker registration ───────────────────────────────────────────────

    def register_speaker(self, device_id: int, name: str) -> None:
        self._speaker_names[device_id]   = name
        self._device_active[device_id]   = True
        self._silence_streak[device_id]  = 0
        self._noise_gate[device_id]      = MIN_ENERGY
        self._device_language[device_id] = None
        self._language_streak[device_id] = 0

        # Fill first available exclusive slot; otherwise shared
        for slot_idx in range(EXCLUSIVE_SLOTS):
            if self._exclusive_slots[slot_idx] is None:
                self._exclusive_slots[slot_idx] = device_id
                self._device_pool[device_id]    = slot_idx
                task = asyncio.create_task(
                    self._exclusive_flush_loop(device_id, slot_idx),
                    name=f"flush_exc{slot_idx}_{device_id}",
                )
                self._exclusive_flush_tasks[device_id] = task
                logger.info(f"Registered: {name} (device {device_id}) → exclusive slot {slot_idx}")
                return

        self._device_pool[device_id] = SHARED_POOL
        logger.info(f"Registered: {name} (device {device_id}) → shared pool")

    def unregister_speaker(self, device_id: int) -> None:
        pool_idx = self._device_pool.pop(device_id, None)
        if pool_idx is not None and pool_idx < EXCLUSIVE_SLOTS:
            self._exclusive_slots[pool_idx] = None
            if task := self._exclusive_flush_tasks.pop(device_id, None):
                task.cancel()
        for d in (self._speaker_names, self._device_active, self._silence_streak,
                  self._noise_gate, self._device_language, self._language_streak):
            d.pop(device_id, None)

    def update_speaker_name(self, device_id: int, name: str) -> None:
        self._speaker_names[device_id] = name

    def set_noise_gate(self, device_id: int, value: float) -> None:
        self._noise_gate[device_id] = max(0.001, min(0.1, value))
        logger.info(f"Device {device_id} noise gate → {value:.4f}")

    # ── Pool assignment (called from API) ─────────────────────────────────

    def assign_exclusive(self, device_id: int, slot_idx: int) -> bool:
        """Assign device to an exclusive slot. Displaces current occupant to shared."""
        if slot_idx < 0 or slot_idx >= EXCLUSIVE_SLOTS:
            return False
        if device_id not in self._device_pool:
            return False

        # Evict current occupant → shared
        occupant = self._exclusive_slots[slot_idx]
        if occupant is not None and occupant != device_id:
            if task := self._exclusive_flush_tasks.pop(occupant, None):
                task.cancel()
            self._device_pool[occupant] = SHARED_POOL
            self._exclusive_slots[slot_idx] = None
            logger.info(f"Device {occupant} displaced from slot {slot_idx} → shared")

        # Remove device from its current slot if exclusive
        old_pool = self._device_pool.get(device_id)
        if old_pool is not None and old_pool < EXCLUSIVE_SLOTS:
            self._exclusive_slots[old_pool] = None
            if task := self._exclusive_flush_tasks.pop(device_id, None):
                task.cancel()

        # Assign
        self._exclusive_slots[slot_idx] = device_id
        self._device_pool[device_id]    = slot_idx
        task = asyncio.create_task(
            self._exclusive_flush_loop(device_id, slot_idx),
            name=f"flush_exc{slot_idx}_{device_id}",
        )
        self._exclusive_flush_tasks[device_id] = task
        logger.info(f"Device {device_id} → exclusive slot {slot_idx}")
        return True

    def unassign_exclusive(self, device_id: int) -> bool:
        """Move device from exclusive slot to shared pool."""
        pool_idx = self._device_pool.get(device_id)
        if pool_idx is None or pool_idx >= EXCLUSIVE_SLOTS:
            return False
        self._exclusive_slots[pool_idx] = None
        if task := self._exclusive_flush_tasks.pop(device_id, None):
            task.cancel()
        self._device_pool[device_id] = SHARED_POOL
        logger.info(f"Device {device_id} → shared pool")
        return True

    def get_pool_status(self) -> dict:
        return {
            "exclusive_slots": {str(k): v for k, v in self._exclusive_slots.items()},
            "device_pool":     {str(k): v for k, v in self._device_pool.items()},
            "shared_devices":  [d for d, p in self._device_pool.items() if p == SHARED_POOL],
        }

    # ── Audio ingestion (no-op: buffer managed by app.py) ────────────────

    async def process_audio_chunk(self, device_id: int, pcm: np.ndarray) -> None:
        pass

    # ── Internal helpers ──────────────────────────────────────────────────

    def _handle_silence(self, device_id: int, audio: np.ndarray, buf) -> bool:
        """Returns True if audio is below noise gate (caller should skip transcription)."""
        rms = float(np.sqrt(np.mean(audio ** 2)))
        threshold = self._noise_gate.get(device_id, MIN_ENERGY)
        if rms < threshold:
            streak = self._silence_streak.get(device_id, 0) + 1
            self._silence_streak[device_id] = streak
            if streak >= 60 and self._device_active.get(device_id, True):
                self._device_active[device_id] = False
                if self.on_device_inactive:
                    asyncio.ensure_future(self.on_device_inactive(device_id))
            if streak >= SILENCE_ADVANCE_AFTER:
                pre_roll = int(PRE_ROLL_SECS * SAMPLE_RATE)
                buf.mark_transcribed(max(0, len(audio) - pre_roll))
            return True

        self._silence_streak[device_id] = 0
        if not self._device_active.get(device_id, True):
            self._device_active[device_id] = True
            if self.on_device_active:
                asyncio.ensure_future(self.on_device_active(device_id))
        return False

    # ── Flush loops ───────────────────────────────────────────────────────

    async def _exclusive_flush_loop(self, device_id: int, pool_idx: int) -> None:
        whisper = self._pools[pool_idx]
        while True:
            await asyncio.sleep(FLUSH_INTERVAL)
            if not self.buffer_mgr:
                continue
            buf = self.buffer_mgr.get_buffer(device_id)
            if buf is None:
                continue
            audio = buf.get_pending()
            if audio is None or len(audio) < int(MIN_AUDIO_SECS * SAMPLE_RATE):
                continue
            if self._handle_silence(device_id, audio, buf):
                continue
            if whisper._busy:
                continue
            asyncio.ensure_future(self._transcribe_and_emit(
                audio, len(audio), device_id,
                self._speaker_names.get(device_id, f"Speaker {device_id}"),
                buf, pool_idx,
            ))

    async def _shared_flush_loop(self) -> None:
        rr_index = 0
        while True:
            await asyncio.sleep(FLUSH_INTERVAL)
            if not self.buffer_mgr:
                continue
            shared = [d for d, p in self._device_pool.items() if p == SHARED_POOL]
            if not shared:
                continue
            whisper = self._pools[SHARED_POOL]
            n = len(shared)
            for i in range(n):
                device_id = shared[(rr_index + i) % n]
                buf = self.buffer_mgr.get_buffer(device_id)
                if buf is None:
                    continue
                audio = buf.get_pending()
                if audio is None or len(audio) < int(MIN_AUDIO_SECS * SAMPLE_RATE):
                    continue
                if self._handle_silence(device_id, audio, buf):
                    continue
                if whisper._busy:
                    break  # come back next tick
                rr_index = (rr_index + i + 1) % n
                asyncio.ensure_future(self._transcribe_and_emit(
                    audio, len(audio), device_id,
                    self._speaker_names.get(device_id, f"Speaker {device_id}"),
                    buf, SHARED_POOL,
                ))
                break  # one transcription per tick

    async def _transcribe_and_emit(self, audio, n_samples, device_id, speaker_name, buf, pool_idx):
        whisper  = self._pools[pool_idx]
        segment  = await whisper.transcribe(audio)
        buf.mark_transcribed(n_samples)

        # Language momentum (informational only — never force)
        detected      = whisper.last_detected_language
        detected_prob = whisper.last_detected_language_probability
        current_lang  = self._device_language.get(device_id)
        if detected in ("en", "es"):
            if current_lang is None:
                if detected_prob >= 0.85:
                    self._device_language[device_id] = detected
                    self._language_streak[device_id] = 0
                    logger.info(f"Device {device_id}: language locked to '{detected}' (p={detected_prob:.2f})")
            elif detected == current_lang:
                self._language_streak[device_id] = 0
            elif detected_prob >= 0.85:
                streak = self._language_streak.get(device_id, 0) + 1
                self._language_streak[device_id] = streak
                if streak >= 3:
                    logger.info(f"Device {device_id}: language '{current_lang}'→'{detected}' after {streak} chunks")
                    self._device_language[device_id] = detected
                    self._language_streak[device_id] = 0

        if segment and segment.text.strip():
            segment.device_id    = device_id
            segment.speaker_name = speaker_name
            if self.on_transcript:
                try:
                    await self.on_transcript(segment)
                except Exception as e:
                    logger.error(f"Transcript callback error: {e}")
