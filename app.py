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
from vad_transcriber import VADTranscriptionPipeline, TranscriptSegment
from audio_recorder import AudioBufferManager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL = "large-v3"
CUDA_DEVICE   = "cuda"
COMPUTE_TYPE  = "int8"
MAX_TRANSCRIPT_HISTORY = 50

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Multi-Speaker Transcription API", version="1.0.0")

audio_manager   = PipeWireAudioManager()
pipeline        = VADTranscriptionPipeline(WHISPER_MODEL, CUDA_DEVICE, COMPUTE_TYPE)
buffer_mgr      = AudioBufferManager()     # ← Lossless audio buffer with cursor

connected_clients: Set[WebSocket] = set()
transcript_history: Dict[int, List[dict]] = {}
active_speakers: Dict[int, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_serial(pw_node_name: str) -> str:
    import re
    m = re.search(r'USB_Composite_Device_([A-F0-9]+)-', pw_node_name, re.IGNORECASE)
    if m:
        return m.group(1)[-4:].upper()
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
    # Layer 2: write to WAV file immediately — no drops, no gaps
    buffer_mgr.add_chunk(device_id, pcm)
    # Layer 1: feed transcription pipeline
    await pipeline.process_audio_chunk(device_id, pcm)


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
    devices = sorted(
        [d for d in devices if "alsa_input.usb" in d.pw_node_name.lower()],
        key=lambda d: d.id
    )
    default_names = ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"]
    for i, device in enumerate(devices[:4]):
        name   = default_names[i]
        serial = extract_serial(device.pw_node_name)
        active_speakers[device.id] = {
            "device_id": device.id,
            "name": name,
            "serial": serial,
            "pw_node_name": device.pw_node_name,
            "mac": device.mac_address,
        }
        pipeline.register_speaker(device.id, name)
        buffer_mgr.register_device(device.id, serial)   # ← Layer 2
        await audio_manager.start_capture(device, callback=on_audio_chunk)

    pipeline.buffer_mgr = buffer_mgr
    await pipeline.load_all_whispers()
    logger.info(f"System ready — {len(devices)} device(s) active")
    
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
            device_ids = list(active_speakers.keys())
            if len(device_ids) < 2:
                consecutive_fused = 0
                continue

            energies = []
            for did in device_ids:
                buf = buffer_mgr.get_buffer(did)
                if buf:
                    pending = buf.get_pending()
                    if pending is not None and len(pending) > 0:
                        energies.append(float(np.sqrt(np.mean(pending ** 2))))

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
                            idx    = len(active_speakers)
                            name   = f"Speaker {idx + 1}"
                            serial = extract_serial(device.pw_node_name)
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
                    idx    = len(active_speakers)
                    name   = f"Speaker {idx + 1}"
                    serial = extract_serial(device.pw_node_name)
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
    success = await audio_manager.rename_device(req.device_id, req.new_name)
    if success:
        pipeline.update_speaker_name(req.device_id, req.new_name)
        # buffer_mgr handles names via pipeline
        if req.device_id in active_speakers:
            active_speakers[req.device_id]["name"] = req.new_name
        await broadcast({"type": "speaker_renamed",
                         "data": {"device_id": req.device_id, "name": req.new_name}})
    return {"success": success}


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
    serial = extract_serial(device.pw_node_name)
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
