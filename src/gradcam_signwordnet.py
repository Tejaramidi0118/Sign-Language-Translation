import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from torchvision import transforms

from signwordnet_model import SignWordNet
from sign_synthetic_dataloader import get_synthetic_dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "best_signwordnet.pth"
NUM_CLASSES = 26
OUTPUT_DIR = "gradcam_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_, val_loader, _, label2idx = get_synthetic_dataloaders()
idx2label = {v: k for k, v in label2idx.items()}

model = SignWordNet(num_classes=NUM_CLASSES)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) else ckpt)
model.to(DEVICE)
model.eval()

target_layer = model.backbone.layer4

activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach()

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

def generate_gradcam(input_tensor, class_idx):
    model.zero_grad()
    output = model(input_tensor)
    score = output[0, class_idx]
    score.backward()

    # Global Average Pooling on gradients
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])

    cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(DEVICE)
    for i, w in enumerate(pooled_grads):
        cam += w * activations[0, i, :, :]

    cam = F.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam.cpu().numpy()

count = 0
MAX_IMAGES = 6

for imgs, labels in val_loader:
    for i in range(imgs.size(0)):
        if count >= MAX_IMAGES:
            break

        img = imgs[i:i+1].to(DEVICE)
        label = labels[i].item()

        with torch.enable_grad():
            pred = torch.argmax(model(img), dim=1).item()
            cam = generate_gradcam(img, pred)

        # Convert tensor image to numpy
        img_np = imgs[i].permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = np.uint8(255 * img_np)

        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        filename = f"gradcam_{count}_true_{idx2label[label]}_pred_{idx2label[pred]}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), overlay)

        count += 1

    if count >= MAX_IMAGES:
        break

print(f"Grad-CAM images saved to '{OUTPUT_DIR}/'")
