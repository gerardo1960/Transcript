#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

notify-send "Live Transcription" "Iniciando..." --icon=audio-input-microphone 2>/dev/null || true

systemctl --user restart pipewire wireplumber 2>/dev/null || true
sleep 3

pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 1

# Abrir terminal visible con el servidor (exec bash lo mantiene abierto al detener)
LOOP="while true; do echo '--- Iniciando servidor ---'; bash '$SCRIPT_DIR/start.sh'; echo '--- Servidor detenido, reiniciando en 15s ---'; sleep 15; done"
if command -v gnome-terminal &>/dev/null; then
    gnome-terminal --title="Live Transcription Server" -- bash -c "$LOOP" &
elif command -v x-terminal-emulator &>/dev/null; then
    x-terminal-emulator -T "Live Transcription Server" -e bash -c "$LOOP" &
elif command -v xterm &>/dev/null; then
    xterm -T "Live Transcription Server" -e bash -c "$LOOP" &
fi

# Esperar a que el servidor esté listo y abrir el browser
for i in $(seq 1 30); do
    curl -s http://localhost:8000/api/speakers > /dev/null 2>&1 && break
    sleep 1
done

notify-send "Live Transcription" "¡Listo!" --icon=audio-input-microphone 2>/dev/null || true
firefox --new-window http://localhost:8000/tablet 2>/dev/null || \
    xdg-open http://localhost:8000/tablet 2>/dev/null || true
