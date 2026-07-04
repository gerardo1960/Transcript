#!/usr/bin/env bash
while true; do
    sleep 30
    if ! curl -s --max-time 5 http://localhost:8000/api/speakers > /dev/null 2>&1; then
        echo "$(date): Server not responding — restarting"
        pkill -f "uvicorn app:app"
        sleep 15          # wait for port 8000 to be released
        bash /home/gerard/Transcript/start.sh &
        sleep 60          # grace period for Whisper models to load before next check
    fi
done
