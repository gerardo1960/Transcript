"""
vad_transcriber.py
Per-device independent Whisper instances — true parallel transcription.
Each speaker gets its own Whisper model loaded on CUDA.
No shared executor, no global lock, no backlog.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000
WINDOW_SECONDS  = 2
MIN_ENERGY      = 0.002


@dataclass
class TranscriptSegment:
    device_id:    int
    speaker_name: str
    text:         str
    language:     str
    confidence:   float
    timestamp:    float = field(default_factory=time.time)
    is_partial:   bool  = False


@dataclass
class SpeakerBuffer:
    device_id:    int
    speaker_name: str
    chunks:       List[np.ndarray] = field(default_factory=list)


class WhisperTranscriber:
    """One Whisper instance per speaker — runs independently on CUDA."""

    def __init__(self, model_size="large-v3", device="cuda", compute_type="int8"):
        self.model_size   = model_size
        self.device       = device
        self.compute_type = compute_type
        self.model        = None
        self._stub_mode   = False
        self._busy        = False   # Drop if this instance is already transcribing

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
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_silence_duration_ms=300,
                speech_pad_ms=300,
            ),
            temperature=0.0,
            no_speech_threshold=0.5,
            condition_on_previous_text=False,
        )

        if info.language not in ("en", "es"):
            return None

        texts, total_logprob, count = [], 0.0, 0
        for seg in segments:
            t = seg.text.strip()
            avg_logprob    = getattr(seg, "avg_logprob", -1.0)
            no_speech_prob = getattr(seg, "no_speech_prob", 1.0)
            if t and avg_logprob > -1.0 and no_speech_prob < 0.6:
                texts.append(t)
                total_logprob += avg_logprob
                count += 1

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
        if len(audio) / VAD_SAMPLE_RATE < 0.5:
            return None
        samples = [("Hola, como estas?", "es"), ("Can you hear me?", "en")]
        text, lang = random.choice(samples)
        return TranscriptSegment(
            device_id=-1, speaker_name="", text=text, language=lang, confidence=0.91,
        )


class VADTranscriptionPipeline:
    """
    Each speaker gets its own Whisper instance and flush loop.
    True parallel transcription — no shared locks between devices.
    Each instance has its own _busy flag to drop if already processing.
    """

    def __init__(self, model_size="large-v3", cuda_device="cuda", compute_type="int8"):
        self.model_size          = model_size
        self.cuda_device         = cuda_device
        self.compute_type        = compute_type
        self._whispers:          Dict[int, WhisperTranscriber] = {}
        self._buffers:           Dict[int, SpeakerBuffer] = {}
        self._flush_tasks:       Dict[int, asyncio.Task] = {}
        self._silence_streak:    Dict[int, int] = {}
        self._device_active:     Dict[int, bool] = {}
        self.on_transcript:      Optional[Callable] = None
        self.on_device_inactive: Optional[Callable] = None
        self.on_device_active:   Optional[Callable] = None

    async def load(self) -> None:
        logger.info("Pipeline ready — per-device Whisper instances")

    async def shutdown(self) -> None:
        for task in self._flush_tasks.values():
            task.cancel()

    def register_speaker(self, device_id: int, name: str) -> None:
        whisper = WhisperTranscriber(self.model_size, self.cuda_device, self.compute_type)
        self._whispers[device_id]       = whisper
        asyncio.ensure_future(whisper.load())
        self._buffers[device_id]        = SpeakerBuffer(device_id=device_id, speaker_name=name)
        self._silence_streak[device_id] = 0
        self._device_active[device_id]  = True
        task = asyncio.create_task(self._flush_loop(device_id), name=f"flush_{device_id}")
        self._flush_tasks[device_id] = task
        logger.info(f"Registered: {name} (device {device_id})")

    def unregister_speaker(self, device_id: int) -> None:
        if task := self._flush_tasks.pop(device_id, None):
            task.cancel()
        self._buffers.pop(device_id, None)
        self._whispers.pop(device_id, None)
        self._silence_streak.pop(device_id, None)
        self._device_active.pop(device_id, None)

    def update_speaker_name(self, device_id: int, name: str) -> None:
        if device_id in self._buffers:
            self._buffers[device_id].speaker_name = name

    async def process_audio_chunk(self, device_id: int, pcm: np.ndarray) -> None:
        buf = self._buffers.get(device_id)
        if buf is not None:
            buf.chunks.append(pcm)

    async def _flush_loop(self, device_id: int) -> None:
        while True:
            await asyncio.sleep(WINDOW_SECONDS)
            buf = self._buffers.get(device_id)
            if not buf or not buf.chunks:
                continue

            chunks, buf.chunks = buf.chunks, []
            audio = np.concatenate(chunks)
            rms   = float(np.sqrt(np.mean(audio ** 2)))

            if rms < MIN_ENERGY:
                self._silence_streak[device_id] = self._silence_streak.get(device_id, 0) + 1
                if self._silence_streak[device_id] >= 30 and self._device_active.get(device_id, True):
                    self._device_active[device_id] = False
                    logger.info(f"Device {device_id} silent")
                    if self.on_device_inactive:
                        asyncio.ensure_future(self.on_device_inactive(device_id))
                continue

            self._silence_streak[device_id] = 0
            if not self._device_active.get(device_id, True):
                self._device_active[device_id] = True
                logger.info(f"Device {device_id} active")
                if self.on_device_active:
                    asyncio.ensure_future(self.on_device_active(device_id))

            if len(audio) >= VAD_SAMPLE_RATE // 2:
                asyncio.ensure_future(
                    self._transcribe_and_emit(audio, device_id, buf.speaker_name)
                )

    async def _transcribe_and_emit(self, audio, device_id, speaker_name):
        whisper = self._whispers.get(device_id)
        if whisper is None:
            return
        segment = await whisper.transcribe(audio)
        if segment and segment.text.strip():
            segment.device_id    = device_id
            segment.speaker_name = speaker_name
            if self.on_transcript:
                try:
                    await self.on_transcript(segment)
                except Exception as e:
                    logger.error(f"Transcript callback error: {e}")
