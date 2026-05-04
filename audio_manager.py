"""
audio_manager.py
PipeWire-based multi-channel audio routing for simultaneous Bluetooth microphone capture.
Each Bluetooth device is treated as an independent speaker channel.
"""

import asyncio
import subprocess
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000          # Whisper expects 16kHz
CHANNELS = 1                 # Mono per speaker
CHUNK_DURATION_MS = 30       # 30ms chunks for low-latency VAD
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
BYTES_PER_SAMPLE = 2         # int16


@dataclass
class AudioDevice:
    """Represents a single Bluetooth microphone / audio source."""
    id: int                          # PipeWire node ID
    name: str                        # Human-friendly display name
    pw_node_name: str                # PipeWire internal node name
    mac_address: Optional[str] = None
    sample_rate: int = SAMPLE_RATE
    active: bool = False
    process: Optional[asyncio.subprocess.Process] = None
    callback: Optional[Callable] = None
    is_stereo: bool = False          # True for DJI wireless receivers (L+R channels)
    stereo_right_id: Optional[int] = None  # Virtual device ID for the R channel


class PipeWireAudioManager:
    """
    Discovers and manages multiple simultaneous Bluetooth audio sources via PipeWire.
    Each active device streams raw PCM audio to an async callback.

    Usage:
        manager = PipeWireAudioManager()
        devices = await manager.discover_bluetooth_devices()
        await manager.start_capture(device, callback=my_async_callback)
    """

    def __init__(self):
        self.devices: Dict[int, AudioDevice] = {}
        self._capture_tasks: Dict[int, asyncio.Task] = {}

    # ------------------------------------------------------------------ #
    #  Device Discovery                                                    #
    # ------------------------------------------------------------------ #

    async def discover_bluetooth_devices(self) -> List[AudioDevice]:
        """
        Query PipeWire for all available audio source nodes.
        Filters for Bluetooth devices by checking node properties.
        Returns list of AudioDevice instances.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "pw-dump",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(f"pw-dump failed: {stderr.decode()}")
                return []

            nodes = json.loads(stdout.decode())
            devices = []

            for node in nodes:
                if node.get("type") != "PipeWire:Interface:Node":
                    continue

                info = node.get("info", {})
                props = info.get("props", {})

                # Only capture sources (microphones)
                media_class = props.get("media.class", "")
                if "Source" not in media_class:
                    continue

                node_id = node.get("id", -1)
                node_name = props.get("node.name", f"device_{node_id}")
                nick = props.get("node.nick") or props.get("node.description") or node_name
                mac = props.get("api.bluez5.address") or props.get("bluez5.address")

                # Prefer Bluetooth devices, but also allow all sources for testing
                is_bt = "bluez" in node_name.lower() or mac is not None

                is_stereo = "wireless_mic_rx" in node_name.lower()
                device = AudioDevice(
                    id=node_id,
                    name=nick,
                    pw_node_name=node_name,
                    mac_address=mac,
                    active=False,
                    is_stereo=is_stereo,
                    stereo_right_id=node_id + 10000 if is_stereo else None,
                )

                devices.append(device)
                self.devices[node_id] = device
                logger.info(f"Discovered {'BT ' if is_bt else ''}device: {nick} (id={node_id})")

            return devices

        except FileNotFoundError:
            logger.warning("pw-dump not found — using simulated devices for development")
            return self._create_simulated_devices()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pw-dump output: {e}")
            return []

    def _create_simulated_devices(self) -> List[AudioDevice]:
        """Create fake devices for development/testing without real hardware."""
        sims = [
            AudioDevice(id=1, name="Speaker 1 (Mic)", pw_node_name="alsa_input.sim1"),
            AudioDevice(id=2, name="Speaker 2 (Mic)", pw_node_name="alsa_input.sim2"),
            AudioDevice(id=3, name="Speaker 3 (Mic)", pw_node_name="alsa_input.sim3"),
        ]
        for d in sims:
            self.devices[d.id] = d
        return sims

    async def get_active_devices(self) -> List[AudioDevice]:
        """Return only currently capturing devices."""
        return [d for d in self.devices.values() if d.active]

    # ------------------------------------------------------------------ #
    #  Capture Control                                                     #
    # ------------------------------------------------------------------ #

    async def start_capture(
        self,
        device: AudioDevice,
        callback: Callable[[int, np.ndarray], None],
    ) -> bool:
        """
        Start raw PCM capture from a PipeWire node using pw-record.
        Streams CHUNK_SAMPLES-sized int16 numpy arrays to `callback(device_id, pcm_array)`.
        """
        if device.active:
            logger.warning(f"Device {device.name} is already capturing")
            return False

        device.callback = callback
        device.active = True

        task = asyncio.create_task(
            self._capture_loop(device),
            name=f"capture_{device.id}",
        )
        self._capture_tasks[device.id] = task
        logger.info(f"Started capture: {device.name}")
        return True

    async def stop_capture(self, device: AudioDevice) -> None:
        """Stop capture for a specific device."""
        device.active = False
        if task := self._capture_tasks.pop(device.id, None):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if device.process and device.process.returncode is None:
            device.process.terminate()
        logger.info(f"Stopped capture: {device.name}")

    async def stop_all(self) -> None:
        """Stop all active captures."""
        for device in list(self.devices.values()):
            if device.active:
                await self.stop_capture(device)

    # ------------------------------------------------------------------ #
    #  Internal Capture Loop                                               #
    # ------------------------------------------------------------------ #

    async def _capture_loop(self, device: AudioDevice) -> None:
        """
        Launch pw-record for a specific node and feed PCM chunks to the callback.
        Stereo devices (DJI wireless receivers) split L/R into two virtual device IDs.
        Falls back to simulated white-noise audio if pw-record is unavailable.
        """
        n_channels = 2 if device.is_stereo else CHANNELS
        cmd = [
            "pw-record",
            "--target", str(device.id),
            "--rate", str(SAMPLE_RATE),
            "--channels", str(n_channels),
            "--format", "s16",
            "-",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            device.process = process
            bytes_per_chunk = CHUNK_SAMPLES * BYTES_PER_SAMPLE * n_channels

            while device.active:
                raw = await process.stdout.readexactly(bytes_per_chunk)
                if device.is_stereo:
                    interleaved = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    left  = interleaved[0::2]
                    right = interleaved[1::2]
                    if device.callback:
                        await device.callback(device.id, left)
                        if device.stereo_right_id is not None:
                            await device.callback(device.stereo_right_id, right)
                else:
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if device.callback:
                        await device.callback(device.id, pcm)

        except FileNotFoundError:
            logger.warning(f"pw-record unavailable — simulating audio for {device.name}")
            await self._simulate_audio(device)
        except asyncio.IncompleteReadError:
            logger.info(f"Audio stream ended for {device.name}")
        except Exception as e:
            logger.error(f"Capture error for {device.name}: {e}")
        finally:
            device.active = False

    async def _simulate_audio(self, device: AudioDevice) -> None:
        """
        Emit simulated silence + occasional random noise bursts for development.
        Simulates ~1s of speech activity every 5s per device.
        """
        t = 0
        while device.active:
            await asyncio.sleep(CHUNK_DURATION_MS / 1000)
            # Simulate periodic speech bursts
            is_speech = (t % 500) < 100  # "speaking" 20% of the time
            if is_speech:
                pcm = np.random.randn(CHUNK_SAMPLES).astype(np.float32) * 0.1
            else:
                pcm = np.zeros(CHUNK_SAMPLES, dtype=np.float32)

            if device.callback:
                await device.callback(device.id, pcm)
            t += 1

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    async def rename_device(self, device_id: int, new_name: str) -> bool:
        """Rename a device (speaker) by its internal ID."""
        if device_id in self.devices:
            self.devices[device_id].name = new_name
            return True
        return False

    def get_device(self, device_id: int) -> Optional[AudioDevice]:
        if device_id in self.devices:
            return self.devices[device_id]
        # Virtual right-channel ID → physical device
        return self.devices.get(device_id - 10000)
