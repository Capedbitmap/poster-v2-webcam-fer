Poster V2 Simple Python Test

Minimal, reproducible webcam demo for POSTER_V2 facial emotion recognition.

What it does
- Detect the largest face via OpenCV Haar cascade
- Classify emotion with POSTER_V2 (AffectNet 8-class checkpoint)
- Draw a bounding box and a color-coded label under the face

Quick start (macOS/Apple Silicon tested)
- Prereqs: git, Python 3.10–3.13, camera permission for your terminal
- One command setup:
  
  bash scripts/bootstrap.sh

- Run the demo:
  
  ./.venv/bin/python src/webcam_emotion.py

- Quit: press q

Notes
- We use the AffectNet 8-class checkpoint because the 7-class file in the upstream repo hardcodes Windows paths for pretrains. The 8-class variant loads pretrains from a relative path.
- Classes assumed: neutral, happy, sad, surprise, fear, disgust, anger, contempt.
- If you prefer the 7-class variant (e.g., RAF-DB), open an issue or PR—we can switch model and paths cleanly.

Project layout
- src/webcam_emotion.py — minimal webcam script
- scripts/bootstrap.sh — reproducible setup (venv, deps, clone, weights)
- external/POSTER_V2 — upstream repo (ignored by git)
- external/POSTER_V2/models/pretrain — pretrain weights (ignored by git)
- external/POSTER_V2/checkpoint — dataset checkpoint (ignored by git)
- .venv — virtual environment (ignored by git)

Troubleshooting
- Camera permission: enable in System Settings → Privacy & Security → Camera.
- GPU/MPS issues: the script auto-uses Apple MPS if available; otherwise CPU.
- Editor import warnings (VS Code): add these settings if needed:
  {
    "python.analysis.extraPaths": ["external/POSTER_V2"],
    "python.defaultInterpreterPath": ".venv/bin/python"
  }

License
MIT for this wrapper repo. POSTER_V2 itself is MIT-licensed (see its LICENSE in the upstream repo).
