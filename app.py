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
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel

from audio_manager import PipeWireAudioManager, AudioDevice
from vad_transcriber import VADTranscriptionPipeline, TranscriptSegment, EXCLUSIVE_SLOTS
from audio_recorder import AudioBufferManager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL = "large-v3"
CUDA_DEVICE   = "cuda"
COMPUTE_TYPE  = "int8"
MAX_TRANSCRIPT_HISTORY = 50
DEFAULT_GAIN_PCT   = 90
DEFAULT_NOISE_GATE = 0.020

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Multi-Speaker Transcription API", version="1.0.0")

audio_manager   = PipeWireAudioManager()
pipeline        = VADTranscriptionPipeline(WHISPER_MODEL, CUDA_DEVICE, COMPUTE_TYPE)
buffer_mgr      = AudioBufferManager()     # ← Lossless audio buffer with cursor

connected_clients: Set[WebSocket] = set()
transcript_history: Dict[int, List[dict]] = {}
active_speakers: Dict[int, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_serial(pw_node_name: str, bus_path: str = "") -> str:
    import re
    # Use USB port path when available — stable and unique even when USB serial is identical
    if bus_path:
        m = re.search(r'usb-\d+:(\d+(?:\.\d+)*)', bus_path)
        if m:
            port = m.group(1).replace(".", "_")
            return f"PORT{port}"
    m = re.search(r'USB_Composite_Device_([A-F0-9]+)-', pw_node_name, re.IGNORECASE)
    if m:
        return m.group(1)[-4:].upper()
    m = re.search(r'Wireless_Mic_Rx_([A-Z0-9]+)-', pw_node_name, re.IGNORECASE)
    if m:
        return m.group(1)[-6:].upper()
    return pw_node_name[-6:].upper()


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
    entry = {
        "device_id": segment.device_id,
        "speaker_name": segment.speaker_name,
        "text": segment.text,
        "language": segment.language,
        "confidence": round(segment.confidence, 3),
        "timestamp": segment.timestamp,
    }
    history = transcript_history.setdefault(segment.device_id, [])
    history.append(entry)
    if len(history) > MAX_TRANSCRIPT_HISTORY:
        history.pop(0)
    await broadcast({"type": "transcript", "data": entry})
    logger.info(f"[{segment.speaker_name}] ({segment.language}): {segment.text[:80]}")


# ── Audio callback — feeds BOTH layers ───────────────────────────────────────

async def on_audio_chunk(device_id: int, pcm: np.ndarray) -> None:
    buffer_mgr.add_chunk(device_id, pcm)


def _apply_gain(device, gain_pct: int):
    """Apply gain to a PipeWire device via pactl (best-effort, non-blocking)."""
    import subprocess
    try:
        subprocess.run(
            ["pactl", "set-source-volume", device.pw_node_name, f"{gain_pct}%"],
            capture_output=True, timeout=3,
        )
    except Exception:
        pass


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("Loading Whisper model …")
    await pipeline.load()
    pipeline.on_transcript = on_transcript_received

    async def on_device_inactive(device_id: int):
        await broadcast({"type": "speaker_inactive", "data": {"device_id": device_id}})
        logger.info(f"Panel hidden for device {device_id}")

    async def on_device_active(device_id: int):
        await broadcast({"type": "speaker_active", "data": {"device_id": device_id}})
        logger.info(f"Panel shown for device {device_id}")

    pipeline.on_device_inactive = on_device_inactive
    pipeline.on_device_active   = on_device_active

    logger.info("Discovering audio devices …")
    devices = await audio_manager.discover_bluetooth_devices()
    usb_devices = sorted(
        [d for d in devices if "alsa_input.usb" in d.pw_node_name.lower()],
        key=lambda d: d.id
    )
    default_names = ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4",
                     "Speaker 5", "Speaker 6", "Speaker 7", "Speaker 8"]
    name_idx = 0
    for device in usb_devices:
        if name_idx >= len(default_names):
            break
        serial = extract_serial(device.pw_node_name, device.bus_path)
        if device.is_stereo:
            for dev_id, suffix in [(device.id, "L"), (device.stereo_right_id, "R")]:
                if name_idx >= len(default_names):
                    break
                name = default_names[name_idx]
                active_speakers[dev_id] = {
                    "device_id": dev_id,
                    "name": name,
                    "serial": serial + suffix,
                    "pw_node_name": device.pw_node_name,
                    "mac": device.mac_address,
                    "gain_pct": DEFAULT_GAIN_PCT,
                    "noise_gate": DEFAULT_NOISE_GATE,
                }
                pipeline.register_speaker(dev_id, name)
                pipeline.set_noise_gate(dev_id, DEFAULT_NOISE_GATE)
                buffer_mgr.register_device(dev_id, serial + suffix)
                name_idx += 1
            await audio_manager.start_capture(device, callback=on_audio_chunk)
            _apply_gain(device, DEFAULT_GAIN_PCT)
        else:
            name = default_names[name_idx]
            active_speakers[device.id] = {
                "device_id": device.id,
                "name": name,
                "serial": serial,
                "pw_node_name": device.pw_node_name,
                "mac": device.mac_address,
                "gain_pct": DEFAULT_GAIN_PCT,
                "noise_gate": DEFAULT_NOISE_GATE,
            }
            pipeline.register_speaker(device.id, name)
            pipeline.set_noise_gate(device.id, DEFAULT_NOISE_GATE)
            buffer_mgr.register_device(device.id, serial)
            await audio_manager.start_capture(device, callback=on_audio_chunk)
            _apply_gain(device, DEFAULT_GAIN_PCT)
            name_idx += 1

    pipeline.buffer_mgr = buffer_mgr
    logger.info(f"System ready — {len(active_speakers)} speaker(s) active")
    
    asyncio.create_task(hotplug_scanner())
    asyncio.create_task(pipewire_watchdog())


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
                    subprocess.run(["systemctl", "--user", "restart", "pipewire", "wireplumber"])
                    await asyncio.sleep(5)
                    devices = await audio_manager.discover_bluetooth_devices()
                    usb_devices = sorted(
                        [d for d in devices if "alsa_input.usb" in d.pw_node_name.lower()],
                        key=lambda d: d.id
                    )
                    for device in usb_devices:
                        if device.id not in active_speakers:
                            serial = extract_serial(device.pw_node_name, device.bus_path)
                            if device.is_stereo:
                                for dev_id, suffix in [(device.id, "L"), (device.stereo_right_id, "R")]:
                                    name = f"Speaker {len(active_speakers) + 1}"
                                    active_speakers[dev_id] = {
                                        "device_id": dev_id,
                                        "name": name,
                                        "serial": serial + suffix,
                                        "pw_node_name": device.pw_node_name,
                                        "mac": device.mac_address,
                                    }
                                    pipeline.register_speaker(dev_id, name)
                                    buffer_mgr.register_device(dev_id, serial + suffix)
                            else:
                                name = f"Speaker {len(active_speakers) + 1}"
                                active_speakers[device.id] = {
                                    "device_id": device.id,
                                    "name": name,
                                    "serial": serial,
                                    "pw_node_name": device.pw_node_name,
                                    "mac": device.mac_address,
                                }
                                pipeline.register_speaker(device.id, name)
                                buffer_mgr.register_device(device.id, serial)
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
                    serial = extract_serial(device.pw_node_name, device.bus_path)
                    if device.is_stereo:
                        for dev_id, suffix in [(device.id, "L"), (device.stereo_right_id, "R")]:
                            name = f"Speaker {len(active_speakers) + 1}"
                            active_speakers[dev_id] = {
                                "device_id": dev_id,
                                "name": name,
                                "serial": serial + suffix,
                                "pw_node_name": device.pw_node_name,
                                "mac": device.mac_address,
                            }
                            pipeline.register_speaker(dev_id, name)
                            buffer_mgr.register_device(dev_id, serial + suffix)
                            await broadcast({"type": "speaker_added", "data": active_speakers[dev_id]})
                        await audio_manager.start_capture(device, callback=on_audio_chunk)
                        logger.info(f"Hotplug: registered DJI stereo (id={device.id})")
                    else:
                        name = f"Speaker {len(active_speakers) + 1}"
                        active_speakers[device.id] = {
                            "device_id": device.id,
                            "name": name,
                            "serial": serial,
                            "pw_node_name": device.pw_node_name,
                            "mac": device.mac_address,
                        }
                        pipeline.register_speaker(device.id, name)
                        buffer_mgr.register_device(device.id, serial)
                        await audio_manager.start_capture(device, callback=on_audio_chunk)
                        await broadcast({"type": "speaker_added", "data": active_speakers[device.id]})
                        logger.info(f"Hotplug: registered {name} (id={device.id})")
        except Exception as e:
            logger.error(f"Hotplug scan error: {e}")


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(str(html_path))

@app.get("/api/speakers")
async def get_speakers():
    return {"speakers": list(active_speakers.values()), "count": len(active_speakers)}

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
        return {"success": False, "error": "Speaker not found"}
    await audio_manager.rename_device(req.device_id, req.new_name)  # best-effort
    pipeline.update_speaker_name(req.device_id, req.new_name)
    active_speakers[req.device_id]["name"] = req.new_name
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
    serial = extract_serial(device.pw_node_name, device.bus_path)
    active_speakers[device.id] = {
        "device_id": device.id, "name": req.name, "serial": serial,
        "pw_node_name": device.pw_node_name, "mac": device.mac_address,
    }
    pipeline.register_speaker(device.id, req.name)
    buffer_mgr.register_device(device.id, serial)
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
    import subprocess
    device = audio_manager.get_device(req.device_id)
    if not device:
        return {"success": False, "error": "Device not found"}
    gain = max(0, min(100, req.gain_pct))
    try:
        result = subprocess.run(
            ["pactl", "set-source-volume", device.pw_node_name, f"{gain}%"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"Device {req.device_id} gain set to {gain}%")
            if req.device_id in active_speakers:
                active_speakers[req.device_id]["gain_pct"] = gain
            return {"success": True, "gain_pct": gain}
        else:
            return {"success": False, "error": result.stderr.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/set_noise_gate")
async def set_noise_gate(req: NoiseGateRequest):
    """Set per-device noise gate threshold."""
    pipeline.set_noise_gate(req.device_id, req.value)
    if req.device_id in active_speakers:
        active_speakers[req.device_id]["noise_gate"] = req.value
    return {"success": True, "value": req.value}

@app.get("/api/get_gain/{device_id}")
async def get_gain(device_id: int):
    """Get current microphone gain from PipeWire."""
    import subprocess, re
    device = audio_manager.get_device(device_id)
    if not device:
        return {"success": False, "error": "Device not found"}
    try:
        result = subprocess.run(
            ["pactl", "get-source-volume", device.pw_node_name],
            capture_output=True, text=True
        )
        m = re.search(r'(\d+)%', result.stdout)
        gain = int(m.group(1)) if m else 100
        return {"success": True, "gain_pct": gain}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Pool Assignment ───────────────────────────────────────────────────────────

@app.get("/api/pool_status")
async def get_pool_status():
    return pipeline.get_pool_status()

class AssignExclusiveRequest(BaseModel):
    device_id: int
    slot_idx:  int

@app.post("/api/assign_exclusive")
async def assign_exclusive(req: AssignExclusiveRequest):
    success = pipeline.assign_exclusive(req.device_id, req.slot_idx)
    if success:
        await broadcast({"type": "pool_changed", "data": pipeline.get_pool_status()})
    return {"success": success}

class UnassignRequest(BaseModel):
    device_id: int

@app.post("/api/unassign_exclusive")
async def unassign_exclusive(req: UnassignRequest):
    success = pipeline.unassign_exclusive(req.device_id)
    if success:
        await broadcast({"type": "pool_changed", "data": pipeline.get_pool_status()})
    return {"success": success}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    logger.info(f"Client connected ({len(connected_clients)} total)")

    await ws.send_text(json.dumps({
        "type": "init",
        "data": {"speakers": list(active_speakers.values()), "history": transcript_history},
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
