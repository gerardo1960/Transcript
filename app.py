"""
app.py
FastAPI server — orchestrates audio capture, transcription, and real-time
WebSocket broadcast to the touch-screen frontend.

Two parallel layers:
  Layer 1: Real-time transcription via Whisper (2s windows)
  Layer 2: Continuous lossless WAV recording (no gaps, no drops)
"""

import asyncio
import json
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel

from audio_manager import PipeWireAudioManager, AudioDevice
from vad_transcriber import VADTranscriptionPipeline, TranscriptSegment
from audio_recorder import AudioBufferManager

# ── Logging ───────────────────────────────────────────────────────────────────
from logging.handlers import RotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            "server.log", mode="a", encoding="utf-8",
            maxBytes=10 * 1024 * 1024,  # 10 MB por archivo
            backupCount=5,              # conserva server.log.1 … server.log.5
        ),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL = "large-v3"
CUDA_DEVICE   = "cuda"
COMPUTE_TYPE  = "int8"
MAX_TRANSCRIPT_HISTORY = 50
DEFAULT_GAIN_PCT   = 100   # AGC off: full passthrough, no hardware compression to compensate for
DEFAULT_NOISE_GATE = 0.010  # AGC off: real noise floor ~0.003–0.005; 0.010 catches soft speech

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Multi-Speaker Transcription API", version="1.0.0")

audio_manager   = PipeWireAudioManager()
pipeline        = VADTranscriptionPipeline(WHISPER_MODEL, CUDA_DEVICE, COMPUTE_TYPE)
buffer_mgr      = AudioBufferManager()     # ← Lossless audio buffer with cursor

connected_clients: Set[WebSocket] = set()
transcript_history: Dict[int, List[dict]] = {}
active_speakers: Dict[int, dict] = {}

# Live audio energy per device — updated every 30ms chunk, exponential decay
# Half-life ≈ 3 s so energy stays relevant across brief pauses between phrases
_live_rms: Dict[int, float] = {}
LIVE_RMS_DECAY = 0.993   # per 30 ms chunk → 0.993^100 ≈ 0.50 after 3 s


# ── Channel naming (sequential, persistent) ───────────────────────────────────

PORT_MAP_FILE = Path(__file__).parent / "port_map.json"
_port_map: Dict[str, int] = {}

def _load_port_map() -> None:
    global _port_map
    try:
        if PORT_MAP_FILE.exists():
            _port_map = json.loads(PORT_MAP_FILE.read_text())
    except Exception:
        _port_map = {}

def _save_port_map() -> None:
    try:
        PORT_MAP_FILE.write_text(json.dumps(_port_map, indent=2))
    except Exception as e:
        logger.warning(f"Could not save port_map: {e}")

def _channel_number(bus_path: str) -> int:
    """Stable sequential number (1-based) for a USB bus-path, persisted across reboots."""
    if bus_path in _port_map:
        return _port_map[bus_path]
    num = max(_port_map.values(), default=0) + 1
    _port_map[bus_path] = num
    _save_port_map()
    logger.info(f"Assigned channel {num} to {bus_path!r}")
    return num

def _channel_label(bus_path: str, pw_node_name: str = "", suffix: str = "") -> str:
    """Short label: '1L', '2R', '3', etc. Falls back to node tail if no bus_path."""
    if bus_path:
        return str(_channel_number(bus_path)) + suffix
    return pw_node_name[-4:].upper() + suffix

# ── Centralized UI config (names, profiles, bench, gains, gates) ──────────────

UI_CONFIG_FILE = Path(__file__).parent / "ui_config.json"
_ui_config: dict = {}

def _load_ui_config() -> None:
    global _ui_config
    defaults: dict = {"saved_names": [], "profiles": [], "bench": [], "gains": {}, "gates": {}, "name_aliases": {}}
    try:
        if UI_CONFIG_FILE.exists():
            loaded = json.loads(UI_CONFIG_FILE.read_text(encoding="utf-8"))
            _ui_config = {**defaults, **loaded}
            return
    except Exception as e:
        logger.warning(f"Could not load ui_config: {e}")
    _ui_config = defaults

def _save_ui_config() -> None:
    try:
        tmp = UI_CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_ui_config, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(UI_CONFIG_FILE)
    except Exception as e:
        logger.warning(f"Could not save ui_config: {e}")


def _sorted_speakers() -> List[dict]:
    """Active speakers sorted by channel label for consistent grid ordering."""
    def _key(sp):
        m = re.match(r'^(\d+)(.*)', sp.get("serial", sp.get("name", "")))
        return (int(m.group(1)), m.group(2)) if m else (999, "")
    return sorted(active_speakers.values(), key=_key)


# ── Voice self-identification ────────────────────────────────────────────────

_IDENTIFY_RE = re.compile(
    r'mi\s+nombre\s+es\s+([a-zà-ÿ\w]+)'
    r'|my\s+name\s+is\s+([a-zà-ÿ\w]+)',
    re.IGNORECASE,
)

def _extract_self_id(text: str) -> Optional[str]:
    m = _IDENTIFY_RE.search(text)
    if not m:
        return None
    return (m.group(1) or m.group(2)).strip()

def _resolve_alias(raw_name: str) -> tuple[str, bool]:
    """Returns (canonical_name, was_known_alias). False = unknown name, just capitalized."""
    raw_lower = raw_name.lower()
    for canonical, variants in _ui_config.get("name_aliases", {}).items():
        if raw_lower == canonical.lower():
            return canonical, True
        if raw_lower in [v.lower() for v in (variants or [])]:
            return canonical, True
    return raw_name.capitalize(), False

_BELL_WAV        = "/usr/share/sounds/sound-icons/glass-water-1.wav"
_PIPER_BIN       = str(Path.home() / ".local/bin/piper")
_PIPER_MODEL_EN  = str(Path.home() / ".local/share/piper-voices/en_US-lessac-high.onnx")
_last_announce_t : float = 0.0          # cooldown: evitar superposición de anuncios
_ANNOUNCE_COOLDOWN = 4.0                # segundos mínimos entre anuncios

async def _announce_registration(canonical_name: str, say_name: bool = True) -> None:
    global _last_announce_t
    now = time.monotonic()
    if now - _last_announce_t < _ANNOUNCE_COOLDOWN:
        logger.info(f"[ANNOUNCE] skipped (cooldown) for '{canonical_name}'")
        return
    _last_announce_t = now
    logger.info(f"[ANNOUNCE] iniciando para '{canonical_name}' say_name={say_name}")
    try:
        # Campanillazo — siempre suena
        bell = await asyncio.create_subprocess_exec(
            "paplay", "--volume=65536", _BELL_WAV,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, bell_err = await bell.communicate()
        logger.info(f"[ANNOUNCE] bell rc={bell.returncode}")

        # TTS "Welcome [nombre]" — solo si el nombre está en la lista de aliases
        if say_name:
            proc = await asyncio.create_subprocess_exec(
                _PIPER_BIN, "--model", _PIPER_MODEL_EN, "--output-raw",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            raw, _ = await proc.communicate(f"Welcome, {canonical_name}".encode())
            player = await asyncio.create_subprocess_exec(
                "paplay", "--raw", "--rate=22050", "--format=s16le", "--channels=1", "--volume=65536",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, play_err = await player.communicate(raw)
            logger.info(f"[ANNOUNCE] paplay rc={player.returncode}")
    except Exception as exc:
        logger.warning(f"[ANNOUNCE] error: {exc}")


# ── Crosstalk / Bleed Suppressor ─────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]")

class CrosstalkSuppressor:
    """
    Suppress transcripts that are mic bleed from a louder neighbour.
    All operations are O(k) where k = recent segment count (~2-8 typical).
    Uses set intersection — microsecond-level, never blocks the event loop.

    Suppresses segment B when source A is louder AND:
      - Word overlap >30% (normal case), OR
      - Source known only via live_rms and is 3× louder (no words committed yet), OR
      - Short/near-silent bleed with massive spike (edge case).
    Double-talk is preserved: different words from similar-volume mics are kept.
    Tuned for DJI Mic Mini with AGC/Clipping Control disabled (linear audio).
    """

    WINDOW_SECS          = 12.0  # history window for committed segments
    VOLUME_RATIO         = 2.5   # room acoustics give 2–10× gap; 2.5× avoids suppressing soft new speakers
    SAME_RECEIVER_RATIO  = 1.1   # DJI L/R channels on same receiver: hardware cross-talk ratio ~1.1–1.2×
    OVERLAP_THRESHOLD    = 0.20  # >20 % of weaker mic's content words must match
    LIVE_ONLY_RATIO      = 6.0   # live-only (no committed words yet) — needs large gap to suppress without word check
    SHORT_WORD_LIMIT     = 3     # segments with ≤3 words get the short-bleed check
    NOISE_FLOOR          = 0.005 # no AGC means hardware noise floor is stable and quieter
    SPIKE_RATIO          = 5.0   # linear audio → source/bleed ratio at near-silence is larger

    def __init__(self):
        # Each entry: (arrival_time, device_id, rms, content_word_set)
        self._recent: deque = deque()

    @staticmethod
    def _same_receiver(dev_a: int, dev_b: int) -> bool:
        """True when two device IDs share the same physical DJI receiver (differ by 10000)."""
        return abs(dev_a - dev_b) == 10000

    @staticmethod
    def _content_words(text: str) -> frozenset:
        """Strip punctuation, lowercase, keep words longer than 3 chars OR pure digits."""
        cleaned = _PUNCT_RE.sub("", text.lower())
        return frozenset(w for w in cleaned.split() if len(w) > 3 or w.isdigit())

    def _purge_old(self, now: float) -> None:
        cutoff = now - self.WINDOW_SECS
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    def _is_bleed(self, rms_b: float, words_b: frozenset, wc_b: int,
                  rms_a: float, words_a: frozenset, volume_ratio: float = None) -> bool:
        """Core bleed test: returns True if B looks like a bleed of A."""
        if volume_ratio is None:
            volume_ratio = self.VOLUME_RATIO
        if rms_a < rms_b * volume_ratio:
            return False
        if wc_b <= self.SHORT_WORD_LIMIT and rms_b < self.NOISE_FLOOR and rms_a >= rms_b * self.SPIKE_RATIO:
            return True
        if words_b and len(words_b & words_a) / len(words_b) > self.OVERLAP_THRESHOLD:
            return True
        return False

    def check_against_history(self, segment, live_rms: dict = None) -> bool:
        """
        Check segment against recently committed segments.
        Aggregates peak RMS and union of content words per device so that
        fragmented transcriptions ("4R" + "en el micrófono" + "esta es una prueba")
        are treated as one combined source rather than three separate weak signals.
        live_rms: current audio energy per device — augments historical RMS so that
        a device actively speaking but not yet committed still registers as dominant.

        When a device is only known via live_rms (no committed words), a stricter
        LIVE_ONLY_RATIO (2×) is used instead of requiring word overlap. This handles
        the race where the source mic's Whisper call finishes after the bleed mic's
        hold period expires.

        Queue-time snapshot: if dominant_peer_rms < rms_volume, this segment was the
        loudest device at the moment it was queued — it is the source, not bleed.
        Skip historical comparison entirely to avoid suppressing consecutive speakers.
        """
        now = time.time()
        self._purge_old(now)

        rms_b     = segment.rms_volume
        words_b   = self._content_words(segment.text)
        wc_b      = len(segment.text.split())
        peer_snap = getattr(segment, "dominant_peer_rms", 0.0)

        # Build per-device peak RMS and accumulated word union from the window.
        # Done before the peer_snap check so dev_words is available for both.
        dev_rms:   Dict[int, float]     = {}
        dev_words: Dict[int, frozenset] = {}
        for _, dev_a, rms_a, words_a in self._recent:
            if dev_a == segment.device_id:
                continue
            if rms_a > dev_rms.get(dev_a, 0.0):
                dev_rms[dev_a] = rms_a
            dev_words[dev_a] = dev_words.get(dev_a, frozenset()) | words_a

        # Augment with live audio energy — catches devices that are loud right now
        # but haven't committed a transcript yet (Whisper still processing)
        if live_rms:
            for dev_id, live in live_rms.items():
                if dev_id != segment.device_id and live > dev_rms.get(dev_id, 0.0):
                    dev_rms[dev_id] = live

        # Peer-snap exemption: if this segment was the dominant device at queue
        # time, it is likely the true source — but verify its words are unique.
        # Late-arriving bleed can appear dominant once the source's live_rms has
        # decayed; in that case the words will match the source's history entries.
        if peer_snap < rms_b:
            if words_b:
                for words_a in dev_words.values():
                    if words_a and len(words_b & words_a) / len(words_b) > self.OVERLAP_THRESHOLD:
                        break  # Words match history → likely late bleed → full check
                else:
                    return False  # Words are unique → genuine new speaker → exempt
            else:
                return False  # No content words → exempt

        for dev_id, rms_a in dev_rms.items():
            ratio   = self.SAME_RECEIVER_RATIO if self._same_receiver(dev_id, segment.device_id) else self.VOLUME_RATIO
            words_a = dev_words.get(dev_id, frozenset())
            if words_a:
                # Source has committed words — use volume + overlap check
                if self._is_bleed(rms_b, words_b, wc_b, rms_a, words_a, ratio):
                    return True
            elif rms_a >= rms_b * self.LIVE_ONLY_RATIO:
                # Source only known via live_rms (no committed words yet).
                # A 3× RMS gap strongly indicates an active dominant speaker.
                return True

        return False

    def record(self, segment) -> None:
        """Commit a segment to the recent history."""
        peer_snap = getattr(segment, "dominant_peer_rms", 0.0)
        rms_b     = segment.rms_volume
        # Skip transition chunks: when a high-rms segment has a significant peer active
        # at queue time, it likely captured the peer's speech as bleed in its tail.
        # Committing it would write the peer's words under the wrong device, causing the
        # peer's own next chunk to be falsely suppressed by HIST.
        HIGH_RMS   = 0.05
        PEER_RATIO = 0.35
        if rms_b > HIGH_RMS and peer_snap > rms_b * PEER_RATIO and peer_snap > self.NOISE_FLOOR:
            logger.info(
                f"[HIST-SKIP] {segment.speaker_name} transition chunk skipped "
                f"(rms={rms_b:.4f}, peer={peer_snap:.4f}, ratio={peer_snap/rms_b:.2f})"
            )
            return
        words = self._content_words(segment.text)
        self._recent.append((time.time(), segment.device_id, segment.rms_volume, words))


crosstalk = CrosstalkSuppressor()

# Hold transcripts for this long before deciding — lets simultaneous Whisper
# results (which finish within ~130ms of each other) all arrive before we compare.
CROSSTALK_HOLD_SECS = 0.80
_pending_segments: list = []   # [(arrival_float, TranscriptSegment)]


# ── WebSocket broadcast ───────────────────────────────────────────────────────

async def broadcast(message: dict) -> None:
    dead = set()
    payload = json.dumps(message)
    for ws in connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)


# ── Transcript callback ───────────────────────────────────────────────────────

async def on_transcript_received(segment: TranscriptSegment) -> None:
    """Queue segment for batch crosstalk evaluation."""
    _pending_segments.append((time.time(), segment))


async def on_transcript_partial_received(segment: TranscriptSegment) -> None:
    """Broadcast instant partial result immediately, bypassing crosstalk suppression."""
    entry = {
        "device_id":    segment.device_id,
        "speaker_name": segment.speaker_name,
        "text":         segment.text,
        "language":     segment.language,
        "confidence":   round(segment.confidence, 3),
        "timestamp":    segment.timestamp,
        "chunk_id":     segment.chunk_id,
    }
    await broadcast({"type": "transcript_partial", "data": entry})


async def on_transcript_remove_received(chunk_id: str, device_id: int) -> None:
    """Tell clients to remove a partial that the large model filtered out."""
    await broadcast({"type": "transcript_remove", "data": {"chunk_id": chunk_id, "device_id": device_id}})


async def _flush_pending_loop() -> None:
    """
    Process pending segments every 80ms.
    Holding CROSSTALK_HOLD_SECS lets all parallel Whisper results (which finish
    within ~150ms of each other) arrive before we compare them against each other.

    Two-phase suppression:
      Phase 1 — within-batch: compare every segment against every other segment
                 that arrived in the same flush window. Handles the common case
                 where 4-8 mics all transcribe the same utterance simultaneously.
      Phase 2 — historical: check survivors against segments committed in the
                 last WINDOW_SECS. Catches staggered bleed where mics happen to
                 flush at different times.
    """
    while True:
        await asyncio.sleep(0.08)
        now    = time.time()
        cutoff = now - CROSSTALK_HOLD_SECS

        ready, still_pending = [], []
        for item in _pending_segments:
            (ready if item[0] <= cutoff else still_pending).append(item)
        _pending_segments.clear()
        _pending_segments.extend(still_pending)
        if not ready:
            continue

        # ── Phase 1: within-batch ─────────────────────────────────────────
        batch_suppressed: set = set()
        # Pre-compute content words once per segment
        meta = [
            (seg, seg.rms_volume,
             CrosstalkSuppressor._content_words(seg.text),
             len(seg.text.split()))
            for _, seg in ready
        ]
        # Aggregate per device across the batch (handles multiple fragments from same mic)
        batch_dev_rms:   Dict[int, float]     = {}
        batch_dev_words: Dict[int, frozenset] = {}
        for seg, rms, words, _ in meta:
            dev = seg.device_id
            if rms > batch_dev_rms.get(dev, 0.0):
                batch_dev_rms[dev] = rms
            batch_dev_words[dev] = batch_dev_words.get(dev, frozenset()) | words

        for i, (seg_b, rms_b, words_b, wc_b) in enumerate(meta):
            if i in batch_suppressed:
                continue
            for dev_a, rms_a in batch_dev_rms.items():
                if dev_a == seg_b.device_id:
                    continue
                ratio   = crosstalk.SAME_RECEIVER_RATIO if crosstalk._same_receiver(dev_a, seg_b.device_id) else crosstalk.VOLUME_RATIO
                words_a = batch_dev_words[dev_a]
                if crosstalk._is_bleed(rms_b, words_b, wc_b, rms_a, words_a, ratio):
                    batch_suppressed.add(i)
                    break

        # ── Phase 2: historical + broadcast ──────────────────────────────
        for i, (seg, rms_b, words_b, wc_b) in enumerate(meta):
            suppressed = (i in batch_suppressed) or crosstalk.check_against_history(seg, _live_rms)
            if suppressed:
                label = "BATCH" if i in batch_suppressed else "HIST"
                logger.info(
                    f"[CROSSTALK-{label}] suppressed {seg.speaker_name} "
                    f"rms={rms_b:.4f}: {seg.text[:60]}"
                )
                if seg.chunk_id:
                    await broadcast({"type": "transcript_remove",
                                     "data": {"chunk_id": seg.chunk_id, "device_id": seg.device_id}})
                continue

            # ── Unnamed mic low-signal filter ─────────────────────────────
            sp_info = active_speakers.get(seg.device_id, {})
            serial  = sp_info.get("serial", "")
            is_unnamed = (sp_info.get("name", "") == serial)  # name == serial → no one assigned
            if is_unnamed and serial:
                gate      = sp_info.get("noise_gate", DEFAULT_NOISE_GATE)
                threshold = gate * 1.5
                is_trigger = _extract_self_id(seg.text) is not None
                if rms_b < threshold and not is_trigger:
                    logger.info(
                        f"[UNNAMED-FILTER] suppressed {seg.speaker_name} "
                        f"rms={rms_b:.4f} < {threshold:.4f}: {seg.text[:60]}"
                    )
                    if seg.chunk_id:
                        await broadcast({"type": "transcript_remove",
                                         "data": {"chunk_id": seg.chunk_id, "device_id": seg.device_id}})
                    continue

            # ── Voice self-identification ─────────────────────────────────
            raw_id = _extract_self_id(seg.text)
            if raw_id is not None and seg.device_id in active_speakers:
                canonical, known = _resolve_alias(raw_id)
                display_name = f"{serial}:{canonical}" if serial else canonical
                logger.info(f"[SELF-ID] dev={seg.device_id} raw='{raw_id}' → '{display_name}' known={known}")
                pipeline.update_speaker_name(seg.device_id, display_name)
                active_speakers[seg.device_id]["name"] = display_name
                seg.speaker_name = display_name
                await broadcast({"type": "speaker_renamed",
                                 "data": {"device_id": seg.device_id, "name": display_name}})
                asyncio.create_task(_announce_registration(canonical, known))

            crosstalk.record(seg)
            entry = {
                "device_id":    seg.device_id,
                "speaker_name": seg.speaker_name,
                "text":         seg.text,
                "language":     seg.language,
                "confidence":   round(seg.confidence, 3),
                "timestamp":    seg.timestamp,
            }
            if seg.chunk_id:
                entry["chunk_id"] = seg.chunk_id
            history = transcript_history.setdefault(seg.device_id, [])
            history.append(entry)
            if len(history) > MAX_TRANSCRIPT_HISTORY:
                history.pop(0)
            event_type = "transcript_update" if seg.chunk_id else "transcript"
            await broadcast({"type": event_type, "data": entry})
            logger.info(f"[{seg.speaker_name}] rms={rms_b:.4f} ({seg.language}): {seg.text[:80]}")


# ── Audio callback — feeds BOTH layers ───────────────────────────────────────

async def on_audio_chunk(device_id: int, pcm: np.ndarray) -> None:
    buffer_mgr.add_chunk(device_id, pcm)
    chunk_rms = float(np.sqrt(np.mean(pcm ** 2)))
    _live_rms[device_id] = max(chunk_rms, _live_rms.get(device_id, 0.0) * LIVE_RMS_DECAY)
    if pipeline:
        await pipeline.process_audio_chunk(device_id, pcm)


async def _apply_gain(device, gain_pct: int):
    """Apply gain to a PipeWire device via pactl (best-effort, non-blocking)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "pactl", "set-source-volume", device.pw_node_name, f"{gain_pct}%",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=3)
    except Exception:
        pass


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("Loading Whisper model …")
    await pipeline.load()
    pipeline.on_transcript         = on_transcript_received
    pipeline.on_transcript_partial = on_transcript_partial_received
    pipeline.on_transcript_remove  = on_transcript_remove_received

    async def on_device_inactive(device_id: int):
        await broadcast({"type": "speaker_inactive", "data": {"device_id": device_id}})
        logger.info(f"Panel hidden for device {device_id}")

    async def on_device_active(device_id: int):
        await broadcast({"type": "speaker_active", "data": {"device_id": device_id}})
        logger.info(f"Panel shown for device {device_id}")

    pipeline.on_device_inactive = on_device_inactive
    pipeline.on_device_active   = on_device_active

    _load_port_map()
    _load_ui_config()
    logger.info("Discovering audio devices …")
    devices = await audio_manager.discover_bluetooth_devices()
    usb_devices = sorted(
        [d for d in devices if "alsa_input.usb" in d.pw_node_name.lower()],
        key=lambda d: d.bus_path or d.pw_node_name,
    )
    for device in usb_devices:
        if device.is_stereo:
            for dev_id, suffix in [(device.id, "L"), (device.stereo_right_id, "R")]:
                name = _channel_label(device.bus_path, device.pw_node_name, suffix)
                gain = _ui_config["gains"].get(name, DEFAULT_GAIN_PCT)
                gate = _ui_config["gates"].get(name, DEFAULT_NOISE_GATE)
                active_speakers[dev_id] = {
                    "device_id": dev_id,
                    "name": name,
                    "serial": name,
                    "pw_node_name": device.pw_node_name,
                    "mac": device.mac_address,
                    "gain_pct": gain,
                    "noise_gate": gate,
                    "bus_path": device.bus_path,
                }
                pipeline.register_speaker(dev_id, name)
                pipeline.set_noise_gate(dev_id, gate)
                buffer_mgr.register_device(dev_id, name)
            # Hardware gain: use L channel value (one physical device)
            name_l = _channel_label(device.bus_path, device.pw_node_name, "L")
            await _apply_gain(device, _ui_config["gains"].get(name_l, DEFAULT_GAIN_PCT))
            await audio_manager.start_capture(device, callback=on_audio_chunk)
        else:
            name = _channel_label(device.bus_path, device.pw_node_name)
            gain = _ui_config["gains"].get(name, DEFAULT_GAIN_PCT)
            gate = _ui_config["gates"].get(name, DEFAULT_NOISE_GATE)
            active_speakers[device.id] = {
                "device_id": device.id,
                "name": name,
                "serial": name,
                "pw_node_name": device.pw_node_name,
                "mac": device.mac_address,
                "gain_pct": gain,
                "noise_gate": gate,
                "bus_path": device.bus_path,
            }
            pipeline.register_speaker(device.id, name)
            pipeline.set_noise_gate(device.id, gate)
            buffer_mgr.register_device(device.id, name)
            await _apply_gain(device, gain)
            await audio_manager.start_capture(device, callback=on_audio_chunk)

    pipeline.buffer_mgr = buffer_mgr
    logger.info(f"System ready — {len(active_speakers)} speaker(s) active")
    
    asyncio.create_task(hotplug_scanner())
    asyncio.create_task(pipewire_watchdog())
    asyncio.create_task(_flush_pending_loop(), name="crosstalk_flush")


@app.on_event("shutdown")
async def shutdown():
    buffer_mgr.stop_all()
    await audio_manager.stop_all()
    await pipeline.shutdown()


# ── PipeWire watchdog ─────────────────────────────────────────────────────────

async def pipewire_watchdog():
    import subprocess
    consecutive_fused = 0
    FUSE_THRESHOLD = 0.98
    FUSE_STRIKES   = 3

    while True:
        await asyncio.sleep(10)
        try:
            buffers = list(pipeline.buffer_mgr._buffers.values()) if pipeline.buffer_mgr else []
            if len(buffers) < 2:
                consecutive_fused = 0
                continue

            energies = []
            for buf in buffers:
                pending = buf.get_pending()
                if pending is not None and len(pending) > 0:
                    energies.append(float(np.sqrt(np.mean(pending[-2400:] ** 2))))

            if len(energies) < 2:
                consecutive_fused = 0
                continue

            max_e = max(energies)
            min_e = min(energies)
            if max_e > 0.001 and (min_e / max_e) > FUSE_THRESHOLD:
                consecutive_fused += 1
                logger.warning(f"PipeWire node fusion detected ({consecutive_fused}/{FUSE_STRIKES})")
                if consecutive_fused >= FUSE_STRIKES:
                    logger.warning("Restarting PipeWire to fix node fusion...")
                    await audio_manager.stop_all()
                    proc = await asyncio.create_subprocess_exec(
                        "systemctl", "--user", "restart", "pipewire", "wireplumber",
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    )
                    await proc.wait()
                    await asyncio.sleep(5)
                    devices = await audio_manager.discover_bluetooth_devices()
                    usb_devices = sorted(
                        [d for d in devices if "alsa_input.usb" in d.pw_node_name.lower()],
                        key=lambda d: d.id
                    )
                    # Match fresh nodes to existing active_speakers by bus_path
                    bp_to_dev_id = {
                        sp.get("bus_path", ""): dev_id
                        for dev_id, sp in active_speakers.items()
                        if sp.get("bus_path")
                    }
                    for device in usb_devices:
                        existing_id = bp_to_dev_id.get(device.bus_path)
                        if existing_id is not None:
                            # Update pw_node_name in the existing speaker record
                            active_speakers[existing_id]["pw_node_name"] = device.pw_node_name
                        await audio_manager.start_capture(device, callback=on_audio_chunk)
                    await broadcast({"type": "pipewire_restarted"})
                    consecutive_fused = 0
            else:
                consecutive_fused = 0
        except Exception as e:
            logger.error(f"Watchdog error: {e}")


# ── Hotplug scanner ───────────────────────────────────────────────────────────

async def hotplug_scanner():
    while True:
        await asyncio.sleep(30)
        try:
            devices = await audio_manager.discover_bluetooth_devices()
            usb_devices = sorted(
                [d for d in devices if "alsa_input.usb" in d.pw_node_name.lower()],
                key=lambda d: d.id
            )
            for device in usb_devices:
                if device.id not in active_speakers and not device.active:
                    if device.is_stereo:
                        for dev_id, suffix in [(device.id, "L"), (device.stereo_right_id, "R")]:
                            name = _channel_label(device.bus_path, device.pw_node_name, suffix)
                            active_speakers[dev_id] = {
                                "device_id": dev_id,
                                "name": name,
                                "serial": name,
                                "pw_node_name": device.pw_node_name,
                                "mac": device.mac_address,
                            }
                            pipeline.register_speaker(dev_id, name)
                            buffer_mgr.register_device(dev_id, name)
                            await broadcast({"type": "speaker_added", "data": active_speakers[dev_id]})
                        await audio_manager.start_capture(device, callback=on_audio_chunk)
                        logger.info(f"Hotplug: registered stereo (id={device.id})")
                    else:
                        name = _channel_label(device.bus_path, device.pw_node_name)
                        active_speakers[device.id] = {
                            "device_id": device.id,
                            "name": name,
                            "serial": name,
                            "pw_node_name": device.pw_node_name,
                            "mac": device.mac_address,
                        }
                        pipeline.register_speaker(device.id, name)
                        buffer_mgr.register_device(device.id, name)
                        await audio_manager.start_capture(device, callback=on_audio_chunk)
                        await broadcast({"type": "speaker_added", "data": active_speakers[device.id]})
                        logger.info(f"Hotplug: registered {name} (id={device.id})")
        except Exception as e:
            logger.error(f"Hotplug scan error: {e}")


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/healthz", include_in_schema=False)
async def healthz():
    return Response(content="ok", media_type="text/plain")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
async def serve_mobile():
    html_path = Path(__file__).parent / "static" / "mobile.html"
    return FileResponse(str(html_path))

@app.get("/mobile", response_class=HTMLResponse)
async def serve_mobile_alias():
    return RedirectResponse(url="/")

@app.get("/tablet", response_class=HTMLResponse)
async def serve_tablet():
    html_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(str(html_path))

@app.get("/manual.html", response_class=HTMLResponse)
async def serve_manual():
    html_path = Path(__file__).parent / "static" / "manual.html"
    return FileResponse(str(html_path))

@app.get("/api/speakers")
async def get_speakers():
    return {"speakers": _sorted_speakers(), "count": len(active_speakers)}

@app.get("/api/history/{device_id}")
async def get_history(device_id: int):
    return {"history": transcript_history.get(device_id, [])}

@app.get("/api/devices")
async def get_devices():
    devices = [
        {"id": d.id, "name": d.name, "pw_node_name": d.pw_node_name,
         "mac": d.mac_address, "active": d.active}
        for d in audio_manager.devices.values()
    ]
    return {"devices": devices}



class RenameRequest(BaseModel):
    device_id: int
    new_name: str

@app.post("/api/rename")
async def rename_speaker(req: RenameRequest):
    if req.device_id not in active_speakers:
        logger.warning(f"Rename failed: device {req.device_id} not in active_speakers")
        return {"success": False, "error": "Speaker not found"}
    await audio_manager.rename_device(req.device_id, req.new_name)  # best-effort
    pipeline.update_speaker_name(req.device_id, req.new_name)
    active_speakers[req.device_id]["name"] = req.new_name
    logger.info(f"Renamed device {req.device_id} → '{req.new_name}'")
    await broadcast({"type": "speaker_renamed",
                     "data": {"device_id": req.device_id, "name": req.new_name}})
    return {"success": True}


class AddSpeakerRequest(BaseModel):
    device_id: int
    name: str

@app.post("/api/add_speaker")
async def add_speaker(req: AddSpeakerRequest):
    device = audio_manager.get_device(req.device_id)
    if not device:
        return {"success": False, "error": "Device not found"}
    if device.active:
        return {"success": False, "error": "Device already active"}
    label = _channel_label(device.bus_path, device.pw_node_name)
    active_speakers[device.id] = {
        "device_id": device.id, "name": req.name, "serial": label,
        "pw_node_name": device.pw_node_name, "mac": device.mac_address,
    }
    pipeline.register_speaker(device.id, req.name)
    buffer_mgr.register_device(device.id, label)
    await audio_manager.start_capture(device, callback=on_audio_chunk)
    await broadcast({"type": "speaker_added", "data": active_speakers[device.id]})
    return {"success": True}

@app.post("/api/remove_speaker/{device_id}")
async def remove_speaker(device_id: int):
    device = audio_manager.get_device(device_id)
    if device and device.active:
        await audio_manager.stop_capture(device)
        pipeline.unregister_speaker(device_id)
        buffer_mgr.unregister_device(device_id)
        active_speakers.pop(device_id, None)
        await broadcast({"type": "speaker_removed", "data": {"device_id": device_id}})
        return {"success": True}
    return {"success": False, "error": "Device not found or not active"}

@app.delete("/api/history/{device_id}")
async def clear_history(device_id: int):
    transcript_history.pop(device_id, None)
    await broadcast({"type": "history_cleared", "data": {"device_id": device_id}})
    return {"success": True}


# ── Mic Controls ─────────────────────────────────────────────────────────────

class GainRequest(BaseModel):
    device_id: int
    gain_pct: int

class NoiseGateRequest(BaseModel):
    device_id: int
    value: float    # 0.001-0.5

@app.post("/api/set_gain")
async def set_gain(req: GainRequest):
    """Set microphone input gain via PipeWire/pactl."""
    device = audio_manager.get_device(req.device_id)
    if not device:
        return {"success": False, "error": "Device not found"}
    gain = max(0, min(100, req.gain_pct))
    try:
        proc = await asyncio.create_subprocess_exec(
            "pactl", "set-source-volume", device.pw_node_name, f"{gain}%",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info(f"Device {req.device_id} gain set to {gain}%")
            if req.device_id in active_speakers:
                serial = active_speakers[req.device_id]["serial"]
                active_speakers[req.device_id]["gain_pct"] = gain
                _ui_config.setdefault("gains", {})[serial] = gain
                _save_ui_config()
                await broadcast({"type": "ui_config_updated", "data": _ui_config})
            return {"success": True, "gain_pct": gain}
        else:
            return {"success": False, "error": stderr.decode().strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/set_noise_gate")
async def set_noise_gate(req: NoiseGateRequest):
    """Set per-device noise gate threshold."""
    pipeline.set_noise_gate(req.device_id, req.value)
    if req.device_id in active_speakers:
        serial = active_speakers[req.device_id]["serial"]
        active_speakers[req.device_id]["noise_gate"] = req.value
        _ui_config.setdefault("gates", {})[serial] = req.value
        _save_ui_config()
        await broadcast({"type": "ui_config_updated", "data": _ui_config})
    return {"success": True, "value": req.value}

@app.get("/api/get_gain/{device_id}")
async def get_gain(device_id: int):
    """Get current microphone gain from PipeWire."""
    import re
    device = audio_manager.get_device(device_id)
    if not device:
        return {"success": False, "error": "Device not found"}
    try:
        proc = await asyncio.create_subprocess_exec(
            "pactl", "get-source-volume", device.pw_node_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        m = re.search(r'(\d+)%', stdout.decode())
        gain = int(m.group(1)) if m else 100
        return {"success": True, "gain_pct": gain}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── UI Config ────────────────────────────────────────────────────────────────

@app.get("/api/ui_config")
async def get_ui_config():
    return _ui_config

class UiConfigPatch(BaseModel):
    saved_names: Optional[List[str]]        = None
    profiles:    Optional[list]             = None
    bench:       Optional[List[str]]        = None
    gains:       Optional[Dict[str, int]]   = None
    gates:       Optional[Dict[str, float]] = None

@app.post("/api/ui_config")
async def patch_ui_config(req: UiConfigPatch):
    changed = False
    if req.saved_names is not None:
        _ui_config["saved_names"] = req.saved_names;  changed = True
    if req.profiles is not None:
        _ui_config["profiles"]    = req.profiles;     changed = True
    if req.bench is not None:
        _ui_config["bench"]       = req.bench;        changed = True
    if req.gains is not None:
        _ui_config.setdefault("gains", {}).update(req.gains);  changed = True
    if req.gates is not None:
        _ui_config.setdefault("gates", {}).update(req.gates);  changed = True
    if changed:
        _save_ui_config()
        await broadcast({"type": "ui_config_updated", "data": _ui_config})
    return {"success": True}


@app.get("/api/name-aliases")
async def get_name_aliases():
    return {"name_aliases": _ui_config.get("name_aliases", {})}

class NameAliasesRequest(BaseModel):
    name_aliases: Dict[str, List[str]]

@app.post("/api/name-aliases")
async def set_name_aliases(req: NameAliasesRequest):
    _ui_config["name_aliases"] = req.name_aliases
    _save_ui_config()
    await broadcast({"type": "ui_config_updated", "data": _ui_config})
    return {"success": True}


class ModeRequest(BaseModel):
    mode: str  # "precise" or "instant"

@app.post("/api/mode")
async def set_transcription_mode(req: ModeRequest):
    if req.mode not in ("precise", "instant"):
        raise HTTPException(status_code=400, detail="mode must be 'precise' or 'instant'")
    pipeline.set_instant_mode(req.mode == "instant")
    await broadcast({"type": "mode_changed", "data": {"mode": req.mode}})
    return {"ok": True, "mode": req.mode}



# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    logger.info(f"Client connected ({len(connected_clients)} total)")

    await ws.send_text(json.dumps({
        "type": "init",
        "data": {
            "speakers":  _sorted_speakers(),
            "history":   transcript_history,
            "ui_config": _ui_config,
        },
    }))

    try:
        while True:
            msg  = await ws.receive_text()
            data = json.loads(msg)
            if data.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))
            elif data.get("type") == "clear_speaker":
                device_id = data.get("device_id")
                if device_id:
                    transcript_history.pop(device_id, None)
                    await broadcast({"type": "history_cleared",
                                     "data": {"device_id": device_id}})
    except WebSocketDisconnect:
        connected_clients.discard(ws)
        logger.info(f"Client disconnected ({len(connected_clients)} remaining)")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connected_clients.discard(ws)
