#!/usr/bin/env bash
notify-send "Live Transcription" "Iniciando..." --icon=audio-input-microphone

systemctl --user restart pipewire wireplumber
sleep 5
pkill -f "uvicorn app:app" 2>/dev/null
sleep 1

x-terminal-emulator -e bash -c "
cd /home/gerard/Transcript
source /home/gerard/.venv/transcription/bin/activate
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/gerard/.venv/transcription/lib/python3.14/site-packages/nvidia/cublas/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/gerard/.venv/transcription/lib/python3.14/site-packages/nvidia/cuda_runtime/lib
uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
exec bash
" &

for i in $(seq 1 30); do
    curl -s http://localhost:8000/api/speakers > /dev/null 2>&1 && break
    sleep 1
done

notify-send "Live Transcription" "¡Listo!" --icon=audio-input-microphone
/usr/bin/firefox --new-window http://localhost:8000
