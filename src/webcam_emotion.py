import os
import sys
import time
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Stubs needed to unpickle upstream checkpoints that reference these classes
# in their original training script (__main__). We ignore their contents.
class RecorderMeter1:
    def __init__(self, *args, **kwargs):
        pass

class RecorderMeter:
    def __init__(self, *args, **kwargs):
        pass

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
POSTER_REPO_DIR = os.path.join(PROJECT_ROOT, "external", "POSTER_V2")
POSTER_MODELS_DIR = os.path.join(POSTER_REPO_DIR, "models")
POSTER_CKPT_PATH = os.path.join(POSTER_REPO_DIR, "checkpoint", "poster_v2_affectnet8.pth")

# Add POSTER_V2 repo to import path
if POSTER_REPO_DIR not in sys.path:
    sys.path.insert(0, POSTER_REPO_DIR)

# Prefer MPS on Apple Silicon if available, else CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# AffectNet-8 standard labels (order assumed)
CLASS_NAMES = [
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "anger",
    "contempt",
]

# Color map per emotion (BGR)
EMOTION_COLORS = {
    "neutral": (200, 200, 200),
    "happy": (40, 180, 40),
    "sad": (200, 120, 0),
    "surprise": (0, 200, 255),
    "fear": (180, 0, 180),
    "disgust": (0, 150, 0),
    "anger": (0, 0, 255),
    "contempt": (100, 100, 255),
}


def load_model() -> torch.nn.Module:
    """Load Poster V2 8-class model with pretrains + dataset checkpoint.

    The POSTER_V2 code loads two pretrains from a relative './pretrain' path inside
    models/PosterV2_8cls. To keep things simple and avoid editing vendor code, we
    temporarily chdir into the models directory when instantiating the model.
    """
    # Import here so sys.path is already configured
    from models.PosterV2_8cls import pyramid_trans_expr2

    # Temporarily change cwd so relative './pretrain' paths resolve correctly
    orig_cwd = os.getcwd()
    try:
        os.chdir(POSTER_MODELS_DIR)
        model = pyramid_trans_expr2(img_size=224, num_classes=8)
    finally:
        os.chdir(orig_cwd)

    # Load dataset checkpoint (AffectNet-8)
    # PyTorch >= 2.6 defaults to weights_only=True, which breaks on this checkpoint.
    # We explicitly set weights_only=False since we trust the source (upstream repo checkpoint).
    ckpt = torch.load(POSTER_CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Handle DataParallel checkpoints where keys are prefixed with "module."
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # Be tolerant to missing keys (e.g., some submodules already loaded from pretrains)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys when loading checkpoint: {len(missing)} (showing first 5) -> {missing[:5]}")
    if unexpected:
        print(f"[warn] Unexpected keys when loading checkpoint: {len(unexpected)} (showing first 5) -> {unexpected[:5]}")

    model.eval()
    model.to(DEVICE)
    return model


def preprocess_face(bgr_img: np.ndarray) -> torch.Tensor:
    """Resize to 224, BGR->RGB, to tensor, normalize like ImageNet."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    # HWC -> CHW
    chw = np.transpose(resized, (0, 1, 2)).astype(np.float32)
    chw = chw / 255.0
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    chw = (chw - mean) / std
    chw = np.transpose(chw, (2, 0, 1))  # to CHW
    tensor = torch.from_numpy(chw).unsqueeze(0)  # 1x3x224x224
    return tensor.to(DEVICE)


def pick_largest_face(faces) -> Tuple[int, int, int, int] | None:
    if len(faces) == 0:
        return None
    # faces: list of (x, y, w, h)
    areas = [w * h for (x, y, w, h) in faces]
    idx = int(np.argmax(areas))
    return tuple(faces[idx])


def main():
    # Sanity checks
    if not os.path.exists(POSTER_CKPT_PATH):
        print(f"Checkpoint not found: {POSTER_CKPT_PATH}")
        print("Make sure weights were downloaded successfully.")
        return
    if not os.path.isdir(POSTER_MODELS_DIR):
        print(f"POSTER_V2 models dir not found at: {POSTER_MODELS_DIR}")
        return

    print(f"Loading model on {DEVICE}...")
    model = load_model()
    print("Model loaded.")

    # Face detector (Haar cascade)
    face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')  # type: ignore[attr-defined]
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("Failed to load Haar cascade for face detection.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. If prompted by macOS, please allow camera access.")
        return

    last_emotion = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
            )

            face_box = pick_largest_face(faces)
            if face_box is not None:
                x, y, w, h = face_box

                # Clamp bbox within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = max(1, min(w, frame.shape[1] - x))
                h = max(1, min(h, frame.shape[0] - y))

                roi = frame[y : y + h, x : x + w]
                with torch.no_grad():
                    inp = preprocess_face(roi)
                    logits = model(inp)
                    # Some model variants might return a tuple; ensure tensor
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    probs = F.softmax(logits, dim=1)
                    pred_idx = int(torch.argmax(probs, dim=1).item())
                    emotion = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # Put label
                label = f"{emotion}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_y = y + h + th + 6
                cv2.rectangle(frame, (x, y + h), (x + max(tw + 8, w), label_y + baseline + 4), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    label,
                    (x + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                last_emotion = emotion
            else:
                # Optional hint when no face
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Poster V2 - Emotion (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
