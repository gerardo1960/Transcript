#!/usr/bin/env bash
# install.sh — Instalador Multi-Speaker Live Transcription
# Compatible con Ubuntu 22.04 / 24.04
# Uso: bash install.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/.venv/transcription"
PYTHON_BIN="python3"
SERVICE_FILE="$HOME/.config/systemd/user/transcription.service"
DESKTOP_FILE="$HOME/.local/share/applications/transcription.desktop"

# ── Colores ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "${RED}✗${NC} $*"; exit 1; }
step() { echo -e "\n${YELLOW}▶${NC} $*"; }

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Multi-Speaker Live Transcription — Instalador"
echo "  Directorio: $APP_DIR"
echo "  Usuario:    $USER  ($HOME)"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── 1. Ubuntu mínimo ──────────────────────────────────────────────────────────
step "Verificando Ubuntu..."
if ! grep -qi ubuntu /etc/os-release 2>/dev/null; then
    warn "No se detectó Ubuntu — continuando de todas formas"
fi
ok "Sistema OK"

# ── 2. Paquetes del sistema ───────────────────────────────────────────────────
step "Instalando paquetes del sistema..."
sudo apt-get update -qq
sudo apt-get install -y \
    pipewire pipewire-pulse pipewire-audio-client-libraries \
    wireplumber pipewire-jack \
    python3 python3-pip python3-venv python3-dev \
    libportaudio2 ffmpeg \
    bluetooth bluez bluez-tools \
    curl notify-send 2>/dev/null || true
ok "Paquetes instalados"

# ── 3. PipeWire ───────────────────────────────────────────────────────────────
step "Habilitando PipeWire..."
systemctl --user enable pipewire pipewire-pulse wireplumber 2>/dev/null || true
systemctl --user start  pipewire pipewire-pulse wireplumber 2>/dev/null || true
ok "PipeWire activo"

# ── 4. Bluetooth ──────────────────────────────────────────────────────────────
step "Habilitando Bluetooth..."
sudo systemctl enable bluetooth 2>/dev/null || true
sudo systemctl start  bluetooth 2>/dev/null || true
ok "Bluetooth activo"

# ── 5. Entorno virtual Python ─────────────────────────────────────────────────
step "Creando entorno virtual Python en $VENV_DIR..."
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q
ok "Virtualenv creado"

# ── 6. GPU / CUDA ─────────────────────────────────────────────────────────────
step "Detectando GPU NVIDIA..."
CUDA_AVAILABLE=false
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU detectada: $GPU_INFO"
    CUDA_AVAILABLE=true
else
    warn "nvidia-smi no encontrado — modo CPU (más lento)"
fi

# ── 7. Dependencias Python ────────────────────────────────────────────────────
step "Instalando dependencias Python..."
pip install -r "$APP_DIR/requirements.txt" -q

if [ "$CUDA_AVAILABLE" = true ]; then
    step "Instalando CTranslate2 con soporte CUDA..."
    pip install ctranslate2 --extra-index-url https://download.pytorch.org/whl/cu121 -q \
        || warn "CTranslate2 CUDA falló — prueba manualmente: pip install ctranslate2"
fi
ok "Dependencias instaladas"

# ── 8. Modelo Whisper ─────────────────────────────────────────────────────────
step "Verificando modelo Whisper large-v3..."
MODELS_DIR="$APP_DIR/models"
mkdir -p "$MODELS_DIR"
MODEL_PATH="$MODELS_DIR/models--Systran--faster-whisper-large-v3"

if [ -d "$MODEL_PATH" ]; then
    ok "Modelo encontrado en $MODELS_DIR"
else
    warn "Modelo no encontrado en $MODELS_DIR"
    echo ""
    echo "  Opciones:"
    echo "  [1] Descargar ahora (~3 GB, requiere internet)"
    echo "  [2] Saltar — se descargará al primer arranque"
    echo ""
    read -rp "  Elige [1/2]: " MODEL_CHOICE
    if [ "${MODEL_CHOICE:-2}" = "1" ]; then
        step "Descargando modelo Whisper large-v3..."
        python3 - <<PYEOF
from faster_whisper import WhisperModel
print("  Iniciando descarga (puede tardar varios minutos)...")
WhisperModel("large-v3", download_root="$MODELS_DIR", device="cpu")
print("  Descarga completa.")
PYEOF
        ok "Modelo descargado"
    else
        warn "Modelo omitido — se descargará automáticamente al primer arranque"
    fi
fi

# ── 9. Script de inicio dinámico ──────────────────────────────────────────────
step "Generando script de inicio..."

# Detectar la ruta de los paquetes nvidia dentro del venv (depende de la versión de Python)
NVIDIA_CUBLAS=""
NVIDIA_CUDA=""
PY_VER=$("$VENV_DIR/bin/python3" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES="$VENV_DIR/lib/$PY_VER/site-packages"

if [ -d "$SITE_PACKAGES/nvidia/cublas/lib" ]; then
    NVIDIA_CUBLAS="$SITE_PACKAGES/nvidia/cublas/lib"
fi
if [ -d "$SITE_PACKAGES/nvidia/cuda_runtime/lib" ]; then
    NVIDIA_CUDA="$SITE_PACKAGES/nvidia/cuda_runtime/lib"
fi

cat > "$APP_DIR/start.sh" << STARTEOF
#!/usr/bin/env bash
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "$VENV_DIR/bin/activate"
STARTEOF

if [ -n "$NVIDIA_CUBLAS" ]; then
    echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$NVIDIA_CUBLAS\"" >> "$APP_DIR/start.sh"
fi
if [ -n "$NVIDIA_CUDA" ]; then
    echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$NVIDIA_CUDA\"" >> "$APP_DIR/start.sh"
fi

cat >> "$APP_DIR/start.sh" << STARTEOF
cd "\$SCRIPT_DIR"
echo "Iniciando servidor en http://localhost:8000"
uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
STARTEOF
chmod +x "$APP_DIR/start.sh"
ok "start.sh generado"

# ── 10. Script de lanzamiento con Firefox ─────────────────────────────────────
cat > "$APP_DIR/launch.sh" << LAUNCHEOF
#!/usr/bin/env bash
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

notify-send "Live Transcription" "Iniciando..." --icon=audio-input-microphone 2>/dev/null || true

systemctl --user restart pipewire wireplumber 2>/dev/null || true
sleep 3

pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 1

# Abrir terminal visible con el servidor (exec bash lo mantiene abierto al detener)
if command -v gnome-terminal &>/dev/null; then
    gnome-terminal --title="Live Transcription Server" -- bash -c "bash '\$SCRIPT_DIR/start.sh'; exec bash" &
elif command -v x-terminal-emulator &>/dev/null; then
    x-terminal-emulator -T "Live Transcription Server" -e bash -c "bash '\$SCRIPT_DIR/start.sh'; exec bash" &
elif command -v xterm &>/dev/null; then
    xterm -T "Live Transcription Server" -e bash -c "bash '\$SCRIPT_DIR/start.sh'; exec bash" &
fi

# Esperar a que el servidor esté listo y abrir el browser
for i in \$(seq 1 30); do
    curl -s http://localhost:8000/api/speakers > /dev/null 2>&1 && break
    sleep 1
done

notify-send "Live Transcription" "¡Listo!" --icon=audio-input-microphone 2>/dev/null || true
firefox --new-window http://localhost:8000 2>/dev/null || \
    xdg-open http://localhost:8000 2>/dev/null || true
LAUNCHEOF
chmod +x "$APP_DIR/launch.sh"
ok "launch.sh generado"

# ── 11. Servicio systemd ──────────────────────────────────────────────────────
step "Configurando servicio systemd..."
mkdir -p "$(dirname "$SERVICE_FILE")"
cat > "$SERVICE_FILE" << SVCEOF
[Unit]
Description=Multi-Speaker Live Transcription
After=pipewire.service

[Service]
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
SVCEOF

if [ -n "$NVIDIA_CUBLAS" ]; then
    echo "Environment=LD_LIBRARY_PATH=$NVIDIA_CUBLAS${NVIDIA_CUDA:+:$NVIDIA_CUDA}" >> "$SERVICE_FILE"
fi

cat >> "$SERVICE_FILE" << SVCEOF

[Install]
WantedBy=default.target
SVCEOF

systemctl --user daemon-reload
ok "Servicio systemd configurado"
echo "  Activar al inicio:  systemctl --user enable transcription"
echo "  Iniciar ahora:      systemctl --user start transcription"

# ── 12. Acceso directo del escritorio ─────────────────────────────────────────
step "Creando acceso directo en el escritorio..."
mkdir -p "$(dirname "$DESKTOP_FILE")"
cat > "$DESKTOP_FILE" << DESKEOF
[Desktop Entry]
Version=1.0
Name=Live Transcription
Comment=Multi-Speaker Live Transcription
Exec=bash $APP_DIR/launch.sh
Icon=audio-input-microphone
Terminal=false
Type=Application
Categories=AudioVideo;Audio;
DESKEOF

if [ -d "$HOME/Desktop" ]; then
    cp "$DESKTOP_FILE" "$HOME/Desktop/transcription.desktop"
    chmod +x "$HOME/Desktop/transcription.desktop"
    ok "Acceso directo creado en el escritorio"
else
    ok "Acceso directo creado en aplicaciones"
fi

# ── 13. Aviso sobre port_map.json ─────────────────────────────────────────────
echo ""
warn "port_map.json — REQUIERE RECONFIGURACIÓN MANUAL"
echo ""
echo "  El archivo port_map.json mapea puertos USB físicos a canales."
echo "  En este equipo nuevo los IDs de puerto serán diferentes."
echo ""
echo "  Pasos después del primer arranque:"
echo "  1. Conecta los micrófonos DJI"
echo "  2. Abre http://localhost:8000"
echo "  3. En Ajustes → los micrófonos aparecerán sin canal asignado"
echo "  4. Usa la interfaz para reasignar los canales"
echo ""

# ── Fin ───────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  ¡Instalación completa!"
echo ""
echo "  Iniciar servidor:   bash $APP_DIR/start.sh"
echo "  Lanzar con browser: bash $APP_DIR/launch.sh"
echo "  URL:                http://localhost:8000"
echo "══════════════════════════════════════════════════════════"
echo ""
