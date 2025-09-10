#!/usr/bin/env bash
set -Eeuo pipefail

# Simple, reproducible setup for POSTER_V2 webcam demo
# - Creates .venv if missing
# - Installs pinned requirements
# - Clones POSTER_V2 (if missing)
# - Downloads pretrained weights and checkpoint

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV="$ROOT_DIR/.venv"
REQUIREMENTS="$ROOT_DIR/requirements.txt"
EXTERNAL_DIR="$ROOT_DIR/external"
POSTER_DIR="$EXTERNAL_DIR/POSTER_V2"

# 1) venv
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/pip" install -r "$REQUIREMENTS"

# 2) clone POSTER_V2
mkdir -p "$EXTERNAL_DIR"
if [ ! -d "$POSTER_DIR/.git" ]; then
  git clone https://github.com/Talented-Q/POSTER_V2 "$POSTER_DIR"
fi

# 3) download pretrains + checkpoint
mkdir -p "$POSTER_DIR/models/pretrain" "$POSTER_DIR/checkpoint"
"$VENV/bin/python" -m gdown --fuzzy 'https://drive.google.com/file/d/17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt/view?usp=sharing' -O "$POSTER_DIR/models/pretrain/ir50.pth"
"$VENV/bin/python" -m gdown --fuzzy 'https://drive.google.com/file/d/1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX/view?usp=sharing' -O "$POSTER_DIR/models/pretrain/mobilefacenet_model_best.pth.tar"
"$VENV/bin/python" -m gdown --fuzzy 'https://drive.google.com/file/d/1tdYH12vgWnIWfupuBkP3jmWS0pJtDxvh/view?usp=sharing' -O "$POSTER_DIR/checkpoint/poster_v2_affectnet8.pth"

echo "\nSetup complete. Run the demo with:"
echo "  $VENV/bin/python src/webcam_emotion.py"

