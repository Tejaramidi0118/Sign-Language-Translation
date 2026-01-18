#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from signwordnet_model import SignWordNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYNTHETIC_DIR = "synthetic_dataset"
CHECKPOINT = "signwordnet_synthetic_best.pt"

def load_label_mapping(root_dir):
    classes = sorted(os.listdir(root_dir))
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    return label2idx, idx2label

def load_model(num_classes, checkpoint_path):
    model = SignWordNet(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # NOTE: no normalization used in training, so we keep it that way
    ])

def predict_image(img_path, model, idx2label):
    tfm = get_transform()
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)   # [1,3,224,224]

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]   # [C]

    topk = 5
    top_probs, top_idxs = torch.topk(probs, k=topk)
    top_probs = top_probs.cpu().numpy()
    top_idxs = top_idxs.cpu().numpy()

    print(f"\n🔍 Predictions for: {img_path}")
    for rank, (p, idx) in enumerate(zip(top_probs, top_idxs), start=1):
        label = idx2label[idx]
        print(f"Top-{rank}: {label} ({p*100:.2f}%)")

    best_label = idx2label[top_idxs[0]]
    return best_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image (.jpg/.png)")
    parser.add_argument("--ckpt", type=str, default=CHECKPOINT,
                        help="Path to trained checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("❌ Image not found:", args.image)
        exit(1)

    if not os.path.exists(args.ckpt):
        print("❌ Checkpoint not found:", args.ckpt)
        exit(1)

    label2idx, idx2label = load_label_mapping(SYNTHETIC_DIR)
    num_classes = len(label2idx)
    print(f"Loaded {num_classes} classes.")

    model = load_model(num_classes, args.ckpt)

    best = predict_image(args.image, model, idx2label)
    print(f"\n✅ Final predicted sign: {best}")
