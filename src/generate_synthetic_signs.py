import os
import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance

INPUT_DIR = "processed_roi_dataset"
OUTPUT_DIR = "synthetic_dataset"
TARGET_PER_CLASS = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

def random_augment(img):
    h, w = img.shape[:2]

    # ---- Random rotation
    angle = random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # ---- Random translation
    tx = random.randint(-15, 15)
    ty = random.randint(-15, 15)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # ---- Random zoom
    scale = random.uniform(0.85, 1.15)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    img = cv2.resize(img, (w, h))

    # ---- Random perspective warp
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = pts1 + np.random.uniform(-10,10,(4,2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # ---- PIL enhancements
    pil_img = Image.fromarray(img)

    # Random sharpness
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(random.uniform(0.8,1.5))
    # Random brightness
    pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.7,1.3))
    # Random contrast
    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.7,1.3))

    img = np.array(pil_img)

    # ---- Random blur
    if random.random() < 0.3:
        k = random.choice([3,5])
        img = cv2.GaussianBlur(img, (k,k), 0)

    return img


for cls in tqdm(os.listdir(INPUT_DIR), desc="Generating synthetic data"):
    in_dir = os.path.join(INPUT_DIR, cls)
    if not os.path.isdir(in_dir):
        continue

    out_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)

    images = [f for f in os.listdir(in_dir) if f.endswith(".jpg")]
    real_count = len(images)
    needed = max(0, TARGET_PER_CLASS - real_count)

    for img_name in images:
        src = os.path.join(in_dir, img_name)
        dst = os.path.join(out_dir, img_name)
        if not os.path.exists(dst):
            os.system(f"copy \"{src}\" \"{dst}\" >nul")

    # Generate synthetic images
    for i in range(needed):
        orig = random.choice(images)
        img = cv2.imread(os.path.join(in_dir, orig))
        aug = random_augment(img)

        out_path = os.path.join(out_dir, f"aug_{i}.jpg")
        cv2.imwrite(out_path, aug)

print("Synthetic dataset created at:", OUTPUT_DIR)
