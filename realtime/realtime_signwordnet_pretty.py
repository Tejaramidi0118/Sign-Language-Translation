#!/usr/bin/env python3
"""
realtime_signwordnet_pretty.py

Real-time ISL word prediction with:
 - Mediapipe hand ROI (same style as processed_roi_dataset)
 - EXACT same preprocessing as training/infer_single_image
 - Nice overlay panel
 - Interactive flow:
      - Live mode (shows top-1 label live)
      - Press 'P'  -> freeze & show "PREDICTION" panel (top-3)
      - Press 'R'  -> resume live mode
      - Press 'Q'  -> quit
"""

import os
import time
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import mediapipe as mp

from signwordnet_model import SignWordNet

# ------------- CONFIG -------------
SYNTHETIC_DIR = "synthetic_dataset"
CHECKPOINT    = "signwordnet_synthetic_best.pt"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LIVE_CONF_THRESHOLD = 0.25    # for label in live mode
PRED_PANEL_TOPK      = 3      # show top-k in paused "Prediction" panel
# ----------------------------------

# ---------- label mapping ----------
def load_label_mapping(root_dir):
    classes = sorted(os.listdir(root_dir))
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    return label2idx, idx2label

# ---------- model loading ----------
def load_model(num_classes, checkpoint_path):
    model = SignWordNet(num_classes=num_classes)
    ck = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
    else:
        state = ck
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# ---------- transform (VERY IMPORTANT) ----------
# EXACTLY match eval_tf from sign_synthetic_dataloader:
#   Resize -> ToTensor() only.  NO Normalize.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------- Mediapipe Hands ----------
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45
)

def crop_from_landmarks(frame_bgr, hand_landmarks, scale=1.1, min_side=80):
    """Crop a square ROI around the detected hand."""
    h, w = frame_bgr.shape[:2]
    xs = np.array([lm.x * w for lm in hand_landmarks.landmark])
    ys = np.array([lm.y * h for lm in hand_landmarks.landmark])

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    box_w = x_max - x_min
    box_h = y_max - y_min
    side = max(box_w, box_h, min_side) * scale

    x1 = int(round(cx - side/2.0))
    y1 = int(round(cy - side/2.0))
    x2 = int(round(cx + side/2.0))
    y2 = int(round(cy + side/2.0))

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame_bgr[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)

# ---------- small UI helpers ----------
def draw_text(img, text, org, scale=0.7, color=(255,255,255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_panel(img, title, topk_labels, topk_scores, conf, w_ratio=0.42):
    """
    Draw a right-side panel:
      title
      MAIN WORD (+ low/high confidence)
      confidence value
      top-3 list
      instructions
    """
    h, w = img.shape[:2]
    panel_w = int(w * w_ratio)
    x0 = w - panel_w
    y0 = 0
    # dark background rectangle
    cv2.rectangle(img, (x0, y0), (w, h), (25, 25, 25), thickness=-1)

    margin_x = x0 + 20
    y = 40

    # Title
    draw_text(img, title, (margin_x, y), scale=0.9, color=(180, 255, 120)); y += 35

    if topk_labels:
        best_label = topk_labels[0]
        main_conf  = topk_scores[0] * 100.0

        conf_str = "high confidence" if main_conf >= 60 else "low confidence"
        # Main prediction
        draw_text(img, f"{best_label}", (margin_x, y), scale=1.0, color=(255,255,255)); y += 30
        draw_text(img, f"({conf_str})", (margin_x, y), scale=0.7, color=(200,200,200)); y += 30

        # Confidence line
        draw_text(img, f"Confidence: {main_conf:.1f}%", (margin_x, y), scale=0.7, color=(200,200,200)); y += 30

        # Top-k list
        draw_text(img, "Top-3:", (margin_x, y), scale=0.8, color=(255,200,150)); y += 28
        for i, (lab, sc) in enumerate(zip(topk_labels, topk_scores), start=1):
            draw_text(img, f"{i}. {lab:12s} {sc*100:.1f}%", (margin_x, y),
                      scale=0.7, color=(255,200,150))
            y += 25
    else:
        draw_text(img, "No prediction yet.", (margin_x, y), scale=0.8, color=(200,200,200)); y += 30

    # Instructions at bottom
    y_instr = h - 40
    draw_text(img, "Press 'P' to capture | 'R' live | 'Q' quit",
              (margin_x, y_instr), scale=0.6, color=(180,180,255))

def run_inference(model, crop_bgr, idx2label):
    """Run model on a BGR crop and return top indices & scores."""
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    x = transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    topk = torch.topk(probs, k=5)
    idxs = topk.indices.cpu().numpy()
    vals = topk.values.cpu().numpy()
    labels = [idx2label[int(i)] for i in idxs]
    scores = [float(v) for v in vals]
    return labels, scores

def main():
    # ---- sanity checks ----
    if not os.path.exists(CHECKPOINT):
        print("❌ Checkpoint not found:", CHECKPOINT); return
    if not os.path.exists(SYNTHETIC_DIR):
        print("❌ synthetic_dataset folder not found:", SYNTHETIC_DIR); return

    label2idx, idx2label = load_label_mapping(SYNTHETIC_DIR)
    num_classes = len(label2idx)
    print(f"Loaded {num_classes} classes.")
    model = load_model(num_classes, CHECKPOINT)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam."); return
    print("✅ Webcam opened. 'P' capture | 'R' resume | 'Q' quit")

    mode = "live"   # "live" or "paused"
    cached_topk_labels = []
    cached_topk_scores = []
    last_fps_t = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        h, w = display.shape[:2]

        # FPS
        now = time.time()
        dt = now - last_fps_t
        last_fps_t = now
        fps = 0.9*fps + 0.1*(1.0/dt) if dt > 0 else fps

        # Detect hand
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        live_label_str = "No Hand"
        live_conf = 0.0
        live_topk_labels = []
        live_topk_scores = []

        crop_for_panel = None

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            crop, box = crop_from_landmarks(frame, hand_lm, scale=1.15, min_side=90)
            if crop is not None:
                crop_for_panel = crop.copy()
                x1,y1,x2,y2 = box
                cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,255), 2)

                if mode == "live":
                    # run model in live mode
                    live_topk_labels, live_topk_scores = run_inference(model, crop, idx2label)
                    live_label_str = live_topk_labels[0]
                    live_conf = live_topk_scores[0]

                    if live_conf < LIVE_CONF_THRESHOLD:
                        live_label_str = "Unknown"

        # small hand thumbnail
        if crop_for_panel is not None:
            thumb = cv2.resize(crop_for_panel, (140,140))
            display[10:10+140, 10:10+140] = thumb

        # draw live info
        draw_text(display, f"FPS: {fps:.1f}", (10, h-20), scale=0.7, color=(0,255,255))

        if mode == "live":
            draw_text(display, f"Live: {live_label_str}", (10, 30),
                      scale=0.9, color=(80,255,80))
            draw_text(display, "Press 'P' to capture word", (10, 60),
                      scale=0.7, color=(200,200,255))
            # right-side panel with current live top-3 (optional)
            draw_panel(display, "PREDICTION (live)", live_topk_labels[:PRED_PANEL_TOPK],
                       live_topk_scores[:PRED_PANEL_TOPK], live_conf)
        else:
            # paused mode: show cached panel
            draw_text(display, "Live: [PAUSED]", (10, 30),
                      scale=0.9, color=(0,220,255))
            draw_panel(display, "PREDICTION", cached_topk_labels,
                       cached_topk_scores, cached_topk_scores[0] if cached_topk_scores else 0.0)

        cv2.imshow("SignWordNet Real-time", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            # freeze current prediction (if crop + detection available)
            if crop_for_panel is not None:
                labels, scores = run_inference(model, crop_for_panel, idx2label)
                cached_topk_labels = labels[:PRED_PANEL_TOPK]
                cached_topk_scores = scores[:PRED_PANEL_TOPK]
                mode = "paused"
        elif key == ord('r'):
            mode = "live"

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Closed webcam.")

if __name__ == "__main__":
    main()
