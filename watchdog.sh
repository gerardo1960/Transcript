#!/usr/bin/env bash
# Requires 2 consecutive failed health checks before restarting,
# with a 15-second timeout per check — tolerates transient CPU/GPU load spikes.
FAIL=0
while true; do
    sleep 30
    if ! curl -s --max-time 15 http://localhost:8000/healthz > /dev/null 2>&1; then
        FAIL=$((FAIL + 1))
        echo "$(date): Server check failed ($FAIL/2)"
        if [ "$FAIL" -lt 2 ]; then
            continue
        fi
        echo "$(date): Server not responding — restarting"
        FAIL=0
        pkill -f "uvicorn app:app"
        sleep 3
        pkill -9 -f "uvicorn app:app"   # SIGKILL if still alive
        sleep 15                          # wait for VRAM to be released
        bash /home/gerard/Transcript/start.sh &
        sleep 60                          # grace period for Whisper models to load
    else
        FAIL=0
    fi
done
