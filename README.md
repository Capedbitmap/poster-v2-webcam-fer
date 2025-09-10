Poster V2 Simple Python Test

This is a minimal test harness to run facial emotion classification using the POSTER_V2 model with a live webcam feed.

What it does:
- Detects a single face via OpenCV Haar cascade
- Draws a bounding box around the largest face
- Runs POSTER_V2 (AffectNet 8-class checkpoint) to classify the emotion
- Renders the predicted emotion under the face; box/text color varies by emotion

Project layout:
- src/webcam_emotion.py — the minimal webcam script
- external/POSTER_V2 — cloned upstream repository (ignored by git)
- external/POSTER_V2/models/pretrain — downloaded pretrain weights (ignored by git)
- external/POSTER_V2/checkpoint — downloaded dataset checkpoint (ignored by git)
- .venv — Python virtual environment (ignored by git)

Setup (already performed by automation):
- Created a local venv: .venv
- Installed minimal deps in the venv: numpy, opencv-python, torch, torchvision, timm, thop, gdown
- Cloned POSTER_V2 into external/POSTER_V2
- Downloaded pretrains (ir50, mobilefacenet) and the AffectNet-8 POSTER_V2 checkpoint

Run it:
- On macOS you may need to grant camera permission to your terminal.
- Start the webcam demo:
  
  ./.venv/bin/python src/webcam_emotion.py

- Press q to quit.

Notes:
- We use the AffectNet 8-class checkpoint for simplicity because the 7-class model file in the repo has hardcoded absolute Windows paths for pretrain weights. The 8-class version loads pretrains from a relative path.
- The 8 classes assumed are: neutral, happy, sad, surprise, fear, disgust, anger, contempt.
- If you want the 7-class variant instead, we can adapt the script to instantiate the 7-class model and fix the pretrain paths.
