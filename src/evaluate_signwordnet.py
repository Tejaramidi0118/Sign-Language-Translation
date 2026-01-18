import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from signwordnet_model import SignWordNet
from sign_synthetic_dataloader import get_synthetic_dataloaders

# ------------------ CONFIG ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "best_signwordnet.pth"   # change if name differs
NUM_CLASSES = 26                            # set correctly
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------
_, _, test_loader, label2idx = get_synthetic_dataloaders()
idx2label = {v: k for k, v in label2idx.items()}

# ------------------ LOAD MODEL ------------------
model = SignWordNet(num_classes=NUM_CLASSES)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) else ckpt)
model.to(DEVICE)
model.eval()

# ------------------ EVALUATION ------------------
all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total * 100
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# ------------------ CLASSIFICATION REPORT ------------------
report = classification_report(
    all_labels,
    all_preds,
    target_names=[idx2label[i] for i in range(NUM_CLASSES)]
)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print("\nClassification Report:\n")
print(report)

# ------------------ CONFUSION MATRIX ------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=[idx2label[i] for i in range(NUM_CLASSES)],
    yticklabels=[idx2label[i] for i in range(NUM_CLASSES)]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix — SIGNWORDNET")
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print(f"\nConfusion matrix saved to: {cm_path}")
print(f"Evaluation results saved in: {RESULTS_DIR}/")
