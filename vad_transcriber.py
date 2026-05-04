"""
vad_transcriber.py
Cursor-based transcription — zero word loss, minimum latency.
Based on 4-2-2026 version + per-device noise gate.

Flush every 0.5s, MIN_AUDIO_SECS=2.5, beam_size=5, strict EN/ES only.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import numpy as np

from audio_recorder import AudioBufferManager, DeviceAudioBuffer

logger = logging.getLogger(__name__)

SAMPLE_RATE    = 16000
FLUSH_INTERVAL = 0.5
MIN_AUDIO_SECS = 1.5
MIN_ENERGY     = 0.002


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

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._load_sync)
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
            return await loop.run_in_executor(None, lambda: self._transcribe_sync(audio))
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
        finally:
            self._busy = False

    def _transcribe_sync(self, audio: np.ndarray) -> Optional[TranscriptSegment]:
        segments, info = self.model.transcribe(
            audio,
            language=None,
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            temperature=0.0,
            no_speech_threshold=0.5,
            condition_on_previous_text=True,
        )

        # Strict: only EN or ES, never anything else
        if info.language not in ("en", "es"):
            return None

        # Low confidence — probably noise
        if info.language_probability < 0.4:
            return None

        texts, total_logprob, count = [], 0.0, 0
        for seg in segments:
            t              = seg.text.strip()
            avg_logprob    = getattr(seg, "avg_logprob", -1.0)
            no_speech_prob = getattr(seg, "no_speech_prob", 1.0)
            if t and avg_logprob > -1.0 and no_speech_prob < 0.6:
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
        self.model_size          = model_size
        self.cuda_device         = cuda_device
        self.compute_type        = compute_type
        self._whispers:          Dict[int, WhisperTranscriber] = {}
        self._flush_tasks:       Dict[int, asyncio.Task] = {}
        self._speaker_names:     Dict[int, str] = {}
        self._device_active:     Dict[int, bool] = {}
        self._silence_streak:    Dict[int, int] = {}
        self._noise_gate:        Dict[int, float] = {}
        self.buffer_mgr:         Optional[AudioBufferManager] = None
        self.on_transcript:      Optional[Callable] = None
        self.on_device_inactive: Optional[Callable] = None
        self.on_device_active:   Optional[Callable] = None

    async def load(self) -> None:
        logger.info("Pipeline ready — cursor-based, zero word loss")

    async def load_all_whispers(self) -> None:
        for device_id, whisper in self._whispers.items():
            logger.info(f"Loading Whisper for device {device_id}...")
            await whisper.load()
        logger.info(f"All {len(self._whispers)} Whisper instances loaded")

    async def shutdown(self) -> None:
        for task in self._flush_tasks.values():
            task.cancel()

    def register_speaker(self, device_id: int, name: str) -> None:
        whisper = WhisperTranscriber(self.model_size, self.cuda_device, self.compute_type)
        self._whispers[device_id]       = whisper
        self._speaker_names[device_id]  = name
        self._device_active[device_id]  = True
        self._silence_streak[device_id] = 0
        self._noise_gate[device_id]     = MIN_ENERGY
        task = asyncio.create_task(
            self._flush_loop(device_id), name=f"flush_{device_id}"
        )
        self._flush_tasks[device_id] = task
        logger.info(f"Registered: {name} (device {device_id})")

    def unregister_speaker(self, device_id: int) -> None:
        if task := self._flush_tasks.pop(device_id, None):
            task.cancel()
        self._whispers.pop(device_id, None)
        self._speaker_names.pop(device_id, None)
        self._device_active.pop(device_id, None)
        self._silence_streak.pop(device_id, None)
        self._noise_gate.pop(device_id, None)

    def update_speaker_name(self, device_id: int, name: str) -> None:
        self._speaker_names[device_id] = name

    def set_noise_gate(self, device_id: int, value: float) -> None:
        self._noise_gate[device_id] = max(0.001, min(0.1, value))
        logger.info(f"Device {device_id} noise gate set to {value:.4f}")

    async def process_audio_chunk(self, device_id: int, pcm: np.ndarray) -> None:
        if self.buffer_mgr:
            self.buffer_mgr.add_chunk(device_id, pcm)

    async def _flush_loop(self, device_id: int) -> None:
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

            rms = float(np.sqrt(np.mean(audio ** 2)))
            device_min_energy = self._noise_gate.get(device_id, MIN_ENERGY)

            if rms < device_min_energy:
                self._silence_streak[device_id] = self._silence_streak.get(device_id, 0) + 1
                if self._silence_streak[device_id] >= 60 and self._device_active.get(device_id, True):
                    self._device_active[device_id] = False
                    logger.info(f"Device {device_id} silent")
                    if self.on_device_inactive:
                        asyncio.ensure_future(self.on_device_inactive(device_id))
                buf.mark_transcribed(len(audio))
                continue

            self._silence_streak[device_id] = 0
            if not self._device_active.get(device_id, True):
                self._device_active[device_id] = True
                logger.info(f"Device {device_id} active")
                if self.on_device_active:
                    asyncio.ensure_future(self.on_device_active(device_id))

            whisper = self._whispers.get(device_id)
            if whisper is None:
                continue

            if whisper._busy:
                continue

            speaker_name = self._speaker_names.get(device_id, f"Speaker {device_id}")
            n_samples    = len(audio)
            asyncio.ensure_future(
                self._transcribe_and_emit(audio, n_samples, device_id, speaker_name, buf)
            )

    async def _transcribe_and_emit(self, audio, n_samples, device_id, speaker_name, buf):
        whisper = self._whispers.get(device_id)
        if whisper is None:
            buf.mark_transcribed(n_samples)
            return

        segment = await whisper.transcribe(audio)
        buf.mark_transcribed(n_samples)

        if segment and segment.text.strip():
            segment.device_id    = device_id
            segment.speaker_name = speaker_name
            if self.on_transcript:
                try:
                    await self.on_transcript(segment)
                except Exception as e:
                    logger.error(f"Transcript callback error: {e}")
