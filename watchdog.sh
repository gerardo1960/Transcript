#!/usr/bin/env bash
while true; do
    sleep 30
    if ! curl -s --max-time 5 http://localhost:8000/api/speakers > /dev/null 2>&1; then
        echo "$(date): Server not responding — restarting"
        pkill -f "uvicorn app:app"
        sleep 3
        pkill -9 -f "uvicorn app:app"   # SIGKILL if still alive
        sleep 15                          # wait for VRAM to be released
        bash /home/gerard/Transcript/start.sh &
        sleep 60                          # grace period for Whisper models to load
    fi
done
