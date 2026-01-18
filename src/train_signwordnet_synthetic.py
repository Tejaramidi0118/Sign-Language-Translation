import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from signwordnet_model import SignWordNet
from sign_synthetic_dataloader import get_synthetic_dataloaders


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
LR = 1e-4
BATCH = 32
CHECKPOINT = "signwordnet_synthetic_best_normalized.pt"

train_loader, val_loader, test_loader, label2idx, idx2label = get_synthetic_dataloaders(
    batch_size=BATCH, num_workers=0
)

num_classes = len(label2idx)
print("Training on classes:", num_classes)

model = SignWordNet(num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val = 0

for ep in range(1, EPOCHS+1):
    model.train()
    total = 0; correct = 0; loss_sum = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        total += imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()

    train_acc = correct / total
    train_loss = loss_sum / total

    # ---- VALIDATION ----
    model.eval()
    val_correct = 0; val_total = 0; val_loss_sum = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)

            val_loss_sum += loss.item() * imgs.size(0)
            val_total += imgs.size(0)
            val_correct += (out.argmax(1) == labels).sum().item()

    val_acc = val_correct / val_total
    val_loss = val_loss_sum / val_total

    scheduler.step()

    print(f"E{ep:02} | TL={train_loss:.4f} | TA={train_acc:.4f} | "
          f"VL={val_loss:.4f} | VA={val_acc:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), CHECKPOINT)
        print("Saved new BEST model:", best_val)

print("Training finished. Best Val Acc =", best_val)
