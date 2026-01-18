import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = "data/raw_corpus/ISL_CSLRT_Corpus/Frames_Word_Level"
OUTPUT_DIR = "processed_roi_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_roi(image):
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None, None

    hand_landmarks = result.multi_hand_landmarks[0]
    landmarks = [(lm.x * w, lm.y * h, lm.z) for lm in hand_landmarks.landmark]
    landmarks_np = np.array(landmarks)

    x_min = int(np.min(landmarks_np[:, 0])) - 20
    x_max = int(np.max(landmarks_np[:, 0])) + 20
    y_min = int(np.min(landmarks_np[:, 1])) - 20
    y_max = int(np.max(landmarks_np[:, 1])) + 20

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, w)
    y_max = min(y_max, h)

    cropped_hand = image[y_min:y_max, x_min:x_max]
    cropped_hand_resized = cv2.resize(cropped_hand, (224, 224))

    # Normalize landmarks relative to crop
    landmarks_np[:, 0] = (landmarks_np[:, 0] - x_min) / (x_max - x_min)
    landmarks_np[:, 1] = (landmarks_np[:, 1] - y_min) / (y_max - y_min)

    return cropped_hand_resized, landmarks_np

# Go through all images
classes = sorted(os.listdir(INPUT_DIR))
for cls in tqdm(classes, desc="Processing classes"):
    class_path = os.path.join(INPUT_DIR, cls)
    if not os.path.isdir(class_path):
        continue

    out_class_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(out_class_dir, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        name_prefix = Path(img_name).stem

        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            cropped_img, lmks = extract_hand_roi(img)
            if cropped_img is None:
                continue

            # Save cropped image
            out_img_path = os.path.join(out_class_dir, f"{name_prefix}.jpg")
            cv2.imwrite(out_img_path, cropped_img)

            # Save landmarks
            out_lmk_path = os.path.join(out_class_dir, f"{name_prefix}_landmarks.npy")
            np.save(out_lmk_path, lmks)

        except Exception as e:
            print(f"Failed: {img_path} → {e}")

print("Done extracting hand ROIs and landmarks.")
