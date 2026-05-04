#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/.venv/transcription/bin/activate"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gerard/.venv/transcription/lib/python3.14/site-packages/nvidia/cublas/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gerard/.venv/transcription/lib/python3.14/site-packages/nvidia/cuda_runtime/lib

cd "$SCRIPT_DIR"
echo "Starting transcription server at http://localhost:8000"
uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
