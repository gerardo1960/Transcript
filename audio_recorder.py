"""
audio_recorder.py
In-memory per-device audio buffer with a transcription cursor.

Instead of dropping audio when Whisper is busy, all audio is kept in a
rolling buffer. Whisper reads from a cursor — if it's busy, the cursor
stays put and the next read includes the accumulated audio.

This guarantees zero word loss while keeping latency as low as possible.
"""

import logging
from collections import deque
from typing import Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE      = 16000
MAX_BUFFER_SECS  = 60    # Keep last 60s in memory
MAX_PENDING_SECS = 3.0   # Max chunk sent to Whisper at once


class DeviceAudioBuffer:
    """
    Circular in-memory buffer with transcription cursor.
    Audio flows in via add_chunk().
    Whisper reads via get_pending() / mark_transcribed().
    If Whisper is busy, cursor stays — audio accumulates for next read.
    """

    def __init__(self, device_id: int, serial: str):
        self.device_id       = device_id
        self.serial          = serial
        self._chunks         = deque()
        self._total_samples  = 0
        self._cursor_samples = 0
        self._pruned_samples = 0

    def add_chunk(self, pcm: np.ndarray) -> None:
        self._chunks.append(pcm.copy())
        self._total_samples += len(pcm)
        self._prune()

    def get_pending(self) -> Optional[np.ndarray]:
        """Return unprocessed audio up to MAX_PENDING_SECS. None if nothing pending."""
        if self._total_samples <= self._cursor_samples:
            return None

        all_audio     = np.concatenate(list(self._chunks)) if self._chunks else np.array([], dtype=np.float32)
        cursor_offset = max(0, self._cursor_samples - self._pruned_samples)

        if cursor_offset >= len(all_audio):
            return None

        pending     = all_audio[cursor_offset:]
        max_samples = int(MAX_PENDING_SECS * SAMPLE_RATE)
        pending     = pending[:max_samples]

        return pending if len(pending) > 0 else None

    def mark_transcribed(self, n_samples: int) -> None:
        """Advance cursor after successful transcription."""
        self._cursor_samples = min(self._cursor_samples + n_samples, self._total_samples)

    def get_context(self, n_samples: int) -> Optional[np.ndarray]:
        """Return the last n_samples of already-transcribed audio as context."""
        all_audio     = np.concatenate(list(self._chunks)) if self._chunks else np.array([], dtype=np.float32)
        cursor_offset = max(0, self._cursor_samples - self._pruned_samples)
        # Get audio just before cursor
        start = max(0, cursor_offset - n_samples)
        context = all_audio[start:cursor_offset]
        return context if len(context) > 0 else None

    def pending_seconds(self) -> float:
        return max(0, self._total_samples - self._cursor_samples) / SAMPLE_RATE

    def _prune(self) -> None:
        """Remove transcribed audio beyond MAX_BUFFER_SECS."""
        max_samples    = int(MAX_BUFFER_SECS * SAMPLE_RATE)
        total_buffered = sum(len(c) for c in self._chunks)
        while self._chunks and total_buffered > max_samples:
            oldest = self._chunks[0]
            if self._pruned_samples + len(oldest) > self._cursor_samples:
                break
            self._chunks.popleft()
            self._pruned_samples += len(oldest)
            total_buffered      -= len(oldest)


class AudioBufferManager:
    """Manages per-device audio buffers."""

    def __init__(self):
        self._buffers: Dict[int, DeviceAudioBuffer] = {}

    def register_device(self, device_id: int, serial: str) -> None:
        if device_id not in self._buffers:
            self._buffers[device_id] = DeviceAudioBuffer(device_id, serial)
            logger.info(f"Buffer registered: device={device_id} serial={serial}")

    def unregister_device(self, device_id: int) -> None:
        self._buffers.pop(device_id, None)

    def add_chunk(self, device_id: int, pcm: np.ndarray) -> None:
        if buf := self._buffers.get(device_id):
            buf.add_chunk(pcm)

    def get_buffer(self, device_id: int) -> Optional[DeviceAudioBuffer]:
        return self._buffers.get(device_id)

    def stop_all(self) -> None:
        self._buffers.clear()
