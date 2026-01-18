from flask import Flask, Response, request, jsonify
import cv2
import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from collections import deque, Counter

from signwordnet_model import SignWordNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "signwordnet_synthetic_best.pt"
SYNTHETIC_DIR = "synthetic_dataset"

TOPK = 3
LIVE_CONF_THRESHOLD = 0.25

STABILITY_FRAMES = 5
VOTING_WINDOW = 7

LAST_TOP1 = ""
MODE = "live"
FINAL_WORD = ""

app = Flask(__name__)

classes = sorted(os.listdir(SYNTHETIC_DIR))
idx2label = {i: c for i, c in enumerate(classes)}

model = SignWordNet(num_classes=len(classes))
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45
)

def crop_from_landmarks(frame, hand_landmarks, scale=1.15, min_side=90):
    h, w = frame.shape[:2]
    xs = np.array([lm.x * w for lm in hand_landmarks.landmark])
    ys = np.array([lm.y * h for lm in hand_landmarks.landmark])
    cx, cy = xs.mean(), ys.mean()
    side = max(xs.max()-xs.min(), ys.max()-ys.min(), min_side) * scale
    x1 = int(max(cx - side/2, 0))
    y1 = int(max(cy - side/2, 0))
    x2 = int(min(cx + side/2, w))
    y2 = int(min(cy + side/2, h))
    if x2 <= x1 or y2 <= y1:
        return None, None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def draw_panel(img, word, topk, show_header=True):
    h, w = img.shape[:2]
    panel_w = int(w * 0.35)
    img[:, w-panel_w:w] = 245
    x = w - panel_w + 30
    y = 60

    if show_header:
        cv2.putText(img, "PREDICTION", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,30,30), 2, cv2.LINE_AA)
        y += 50
    else:
        y += 10

    if word:
        cv2.putText(img, word, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20,20,20), 3, cv2.LINE_AA)
        y += 40

    if show_header:
        cv2.line(img, (x, y), (w-30, y), (180,180,180), 1)
        y += 35
    else:
        y += 10

    cv2.putText(img, "Top results", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60,60,60), 2, cv2.LINE_AA)
    y += 30
    for i, t in enumerate(topk[:TOPK], 1):
        cv2.putText(img, f"{i}. {t}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40,40,40), 2, cv2.LINE_AA)
        y += 28

def draw_popup(img, word):
    h, w = img.shape[:2]

    # popup size
    box_w = int(w * 0.45)
    box_h = int(h * 0.25)

    cx, cy = w // 2, h // 2
    x1, y1 = cx - box_w//2, cy - box_h//2
    x2, y2 = cx + box_w//2, cy + box_h//2

    # shadow
    cv2.rectangle(img, (x1+5, y1+5), (x2+5, y2+5), (210,210,210), -1)
    # box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), -1)

    title = "Predicted word"
    (tw1, th1), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    tx1 = cx - tw1 // 2
    ty1 = y1 + 45

    cv2.putText(img, title, (tx1, ty1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90,90,90), 2, cv2.LINE_AA)

    cv2.line(img, (x1+40, ty1+15), (x2-40, ty1+15), (180,180,180), 1)

    (tw2, th2), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1.9, 3)
    tx2 = cx - tw2 // 2
    ty2 = ty1 + 70

    cv2.putText(img, word, (tx2, ty2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.9, (20,20,20), 3, cv2.LINE_AA)

def draw_hand_inset(frame, hand_crop, pos="topleft"):
    if hand_crop is None:
        return

    INSET_SIZE = 160
    PADDING = 10
    BORDER = 2

    # Resize crop neatly
    inset = cv2.resize(hand_crop, (INSET_SIZE, INSET_SIZE))

    inset = inset.copy()

    h, w = frame.shape[:2]

    if pos == "topleft":
        x1 = PADDING
        y1 = PADDING + 40 
    elif pos == "topright":
        x1 = w - INSET_SIZE - PADDING
        y1 = PADDING + 40
    else:
        return

    x2 = x1 + INSET_SIZE
    y2 = y1 + INSET_SIZE

    cv2.rectangle(
        frame,
        (x1 - BORDER, y1 - BORDER),
        (x2 + BORDER, y2 + BORDER),
        (255, 255, 255),
        -1
    )

    cv2.rectangle(
        frame,
        (x1 - BORDER, y1 - BORDER),
        (x2 + BORDER, y2 + BORDER),
        (180, 180, 180),
        BORDER
    )

    frame[y1:y2, x1:x2] = inset

def generate():
    global MODE, FINAL_WORD, LAST_TOP1
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=VOTING_WINDOW)
    stable_word = None
    stable_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        topk_labels = []
        crop_for_panel = None

        if MODE == "live":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)
            if results.multi_hand_landmarks:
                crop, box = crop_from_landmarks(frame, results.multi_hand_landmarks[0])
                if crop is not None:
                    x1,y1,x2,y2 = box
                    cv2.rectangle(display, (x1,y1),(x2,y2), (0,180,255), 2)
                    
                    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    x = transform(pil).unsqueeze(0).to(DEVICE)
                    
                    crop_for_panel = crop.copy()

                    with torch.no_grad():
                        probs = F.softmax(model(x), dim=1)[0]

                    topk = torch.topk(probs, k=TOPK)
                    topk_labels = [idx2label[int(i)] for i in topk.indices]

                    LAST_TOP1 = topk_labels[0] if topk_labels else ""

                    if probs[topk.indices[0]] >= LIVE_CONF_THRESHOLD:
                        buffer.append(topk_labels[0])
                        if topk_labels[0] == stable_word:
                            stable_count += 1
                        else:
                            stable_word = topk_labels[0]
                            stable_count = 1
            else:
                LAST_TOP1 = "NO HAND DETECTED"

        if MODE == "paused" and FINAL_WORD:
            draw_popup(display, FINAL_WORD)
        else:
            draw_panel(display, "", topk_labels, show_header=False)

        draw_hand_inset(display, crop_for_panel, pos="topleft")

        ret, jpeg = cv2.imencode(".jpg", display)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return """
    <html>
    <head>
      <style>
        body { margin:0; background:#f3f4f6; font-family:Arial, sans-serif; }
        .controls { position:fixed; top:12px; left:12px; z-index:10; }
        button { padding:9px 14px; margin-right:8px; border:1px solid #ccc; background:white; cursor:pointer; border-radius:6px; }
        button:hover { background:#eee; }
      </style>
    </head>
    <body>
      <div class="controls">
        <button onclick="fetch('/pause')">P – Predict</button>
        <button onclick="fetch('/resume')">R – Resume</button>

        <br><br>

        <input type="file" id="imgInput" accept="image/*">
        <button onclick="uploadImage()">Predict Image</button>
        </div>

        <script>
        function uploadImage() {
        const fileInput = document.getElementById("imgInput");
        if (!fileInput.files.length) {
            alert("Please select an image");
            return;
        }

        const formData = new FormData();
        formData.append("image", fileInput.files[0]);

        fetch("/predict_image", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            console.log("Predicted:", data.predicted_word);
        })
        .catch(err => console.error(err));
        }
        </script>

      <img src="/video_feed" width="100%">
    </body>
    </html>
    """

@app.route("/pause")
def pause():
    global MODE, FINAL_WORD, LAST_TOP1
    FINAL_WORD = LAST_TOP1
    MODE = "paused"
    return "paused"

@app.route("/resume")
def resume():
    global MODE, FINAL_WORD
    MODE = "live"
    FINAL_WORD = ""
    return "live"

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/predict_image", methods=["POST"])
def predict_image():
    global FINAL_WORD, MODE

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    x = transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]

    top1_idx = int(torch.argmax(probs))
    FINAL_WORD = idx2label[top1_idx]

    MODE = "paused"

    return jsonify({
        "predicted_word": FINAL_WORD
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
