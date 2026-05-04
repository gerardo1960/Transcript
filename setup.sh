#!/usr/bin/env bash
# setup.sh — One-shot setup for Multi-Speaker Transcription on Ubuntu 24.04
# Run as: bash setup.sh
set -euo pipefail

VENV_DIR="$HOME/.venv/transcription"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Multi-Speaker Transcription — Setup"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "▶ Installing system packages…"
sudo apt-get update -qq
sudo apt-get install -y \
    pipewire \
    pipewire-pulse \
    pipewire-audio-client-libraries \
    wireplumber \
    pipewire-jack \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libportaudio2 \
    ffmpeg \
    bluetooth \
    bluez \
    bluez-tools

echo "✓ System packages installed"

# ── 2. Bluetooth service ──────────────────────────────────────────────────────
echo "▶ Enabling Bluetooth service…"
sudo systemctl enable bluetooth
sudo systemctl start bluetooth
echo "✓ Bluetooth enabled"

# ── 3. Python virtual environment ─────────────────────────────────────────────
echo "▶ Creating Python virtualenv at $VENV_DIR…"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q

echo "✓ Virtualenv created"

# ── 4. CUDA check ─────────────────────────────────────────────────────────────
echo "▶ Checking CUDA availability…"
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "✓ GPU detected: $GPU_INFO"
    CUDA_AVAILABLE=true
else
    echo "⚠  nvidia-smi not found — will run in CPU mode (slower)"
    CUDA_AVAILABLE=false
fi

# ── 5. Python dependencies ────────────────────────────────────────────────────
echo "▶ Installing Python packages…"
pip install -r "$APP_DIR/requirements.txt" -q

if [ "$CUDA_AVAILABLE" = true ]; then
    echo "▶ Installing CTranslate2 with CUDA support…"
    pip install ctranslate2 --extra-index-url https://download.pytorch.org/whl/cu121 -q || \
        echo "⚠  CTranslate2 CUDA install failed — try manual install"
fi

echo "✓ Python packages installed"

# ── 6. Download Whisper model ─────────────────────────────────────────────────
MODELS_DIR="$APP_DIR/models"
mkdir -p "$MODELS_DIR"
echo "▶ Whisper model will be auto-downloaded on first run to: $MODELS_DIR"
echo "  (large-v3 is ~3GB — ensure disk space is available)"
echo "  To pre-download: python -c \"from faster_whisper import WhisperModel; WhisperModel('large-v3', download_root='$MODELS_DIR')\""

# ── 7. Systemd service (optional) ─────────────────────────────────────────────
SERVICE_FILE="$HOME/.config/systemd/user/transcription.service"
mkdir -p "$(dirname "$SERVICE_FILE")"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Multi-Speaker Live Transcription
After=pipewire.service

[Service]
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

echo "✓ Systemd user service written to: $SERVICE_FILE"
echo "  Enable at login: systemctl --user enable transcription"
echo "  Start now:       systemctl --user start transcription"

# ── 8. Launch script ──────────────────────────────────────────────────────────
cat > "$APP_DIR/start.sh" << 'LAUNCH'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/.venv/transcription/bin/activate"
cd "$SCRIPT_DIR"
echo "Starting transcription server at http://localhost:8000"
uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
LAUNCH
chmod +x "$APP_DIR/start.sh"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Start the server:   ./start.sh"
echo "  Open in browser:    http://localhost:8000"
echo ""
echo "  For fullscreen kiosk mode (touchscreen):"
echo "    chromium-browser --kiosk --app=http://localhost:8000"
echo "══════════════════════════════════════════════════════════"
echo ""
