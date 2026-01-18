import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class SyntheticSignDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def get_synthetic_dataloaders(
    root="synthetic_dataset",
    batch_size=32,
    num_workers=0,
    image_size=224,
    val_ratio=0.15,
    test_ratio=0.15,
):

    # 1. Load ALL images & labels
    classes = sorted(os.listdir(root))
    label2idx = {cls: i for i, cls in enumerate(classes)}
    idx2label = {i: cls for cls, i in label2idx.items()}

    all_samples = []
    for cls in classes:
        folder = os.path.join(root, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                all_samples.append(
                    (os.path.join(folder, fname), label2idx[cls])
                )

    print(f"Loaded {len(all_samples)} images from {len(classes)} classes.")

    # 2. Split train / val / test
    labels = [lbl for _, lbl in all_samples]

    train_samples, temp_samples = train_test_split(
        all_samples,
        test_size=val_ratio + test_ratio,
        stratify=labels,
        random_state=42,
    )

    temp_labels = [lbl for _, lbl in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=42,
    )

    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}")

    # 3. Transforms
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # 4. Create Dataset objects
    # ---------------------------
    train_set = SyntheticSignDataset(train_samples, transform=train_tf)
    val_set = SyntheticSignDataset(val_samples, transform=eval_tf)
    test_set = SyntheticSignDataset(test_samples, transform=eval_tf)

    # 5. Create BALANCED WeightedRandomSampler for TRAIN
    train_labels = [lbl for _, lbl in train_samples]
    counts = Counter(train_labels)

    # weight per sample
    train_weights = torch.DoubleTensor([1.0 / counts[lbl] for lbl in train_labels])

    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )

    # 6. DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, label2idx, idx2label
