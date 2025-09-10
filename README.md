# Poster V2 Webcam Emotion (FER)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10--3.13-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey)

Minimal, reproducible live webcam demo for POSTER_V2 facial expression recognition (FER).

## Features
- Detects the largest face via OpenCV Haar cascade
- Classifies emotion using POSTER_V2 (AffectNet 8-class checkpoint)
- Draws a bounding box and color-coded label under the face
- Fully local: venv + pinned dependencies; no global installs

## Quick Start (macOS/Apple Silicon tested)
- Prereqs: git, Python 3.10–3.13, camera permission for your terminal
- Setup:
  ```bash path=null start=null
  bash scripts/bootstrap.sh
  ```
- Run:
  ```bash path=null start=null
  ./.venv/bin/python src/webcam_emotion.py
  ```
- Quit: press q

## Notes
- We use the AffectNet 8-class checkpoint because the 7-class model in the upstream repo hardcodes Windows paths for pretrains. The 8-class variant loads pretrains from a relative path.
- FER = Facial Expression Recognition. Classes assumed: neutral, happy, sad, surprise, fear, disgust, anger, contempt.
- Prefer another dataset (e.g., RAF-DB 7-class)? Open an issue/PR—we can switch the architecture and paths cleanly.

## Project Layout
- `src/webcam_emotion.py` — minimal webcam script
- `scripts/bootstrap.sh` — reproducible setup (venv, deps, clone, weights)
- `external/POSTER_V2` — upstream repo (ignored by git)
- `external/POSTER_V2/models/pretrain` — pretrain weights (ignored)
- `external/POSTER_V2/checkpoint` — dataset checkpoint (ignored)
- `.venv` — local virtual environment (ignored)

## Reproducibility
- Pinned dependencies in `requirements.txt` (torch, torchvision, timm, thop, numpy, OpenCV, gdown, pillow)
- One-command bootstrap sets up venv, installs pins, clones upstream, downloads weights
- No global installs; everything stays within `.venv`

## Troubleshooting
- Camera permission: System Settings → Privacy & Security → Camera → allow your terminal
- GPU/MPS: the script uses Apple MPS if available; otherwise CPU
- VS Code import hints (optional settings):
  ```json path=null start=null
  {
    "python.analysis.extraPaths": ["external/POSTER_V2"],
    "python.defaultInterpreterPath": ".venv/bin/python"
  }
  ```

## Acknowledgements
- Upstream model: POSTER V2 — https://github.com/Talented-Q/POSTER_V2 (MIT)

## License
- MIT for this wrapper repo (see [LICENSE](LICENSE))
- POSTER_V2 is MIT-licensed in its own repository
