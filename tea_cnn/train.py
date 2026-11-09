"""
Trains a MobileNetV2-based transfer-learning classifier on the combined
tea-leaf-disease dataset produced by load_data.py.

Run after load_data.py: python3 train.py
"""
import csv
import json
import os

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import (
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_DIR,
    DATA_CACHE_DIR,
    HISTORY_CSV_PATH,
    IMG_SIZE,
    LABELS_PATH,
    SEED,
)

BATCH_SIZE = 16
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example["image"].convert("RGB")
        label = example["label"]
        return self.transform(image), label


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, total_correct, total_count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_count += images.size(0)
    return total_loss / total_count, total_correct / total_count


def main():
    torch.manual_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(LABELS_PATH) as f:
        classes = json.load(f)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    train_hf = load_from_disk(os.path.join(DATA_CACHE_DIR, "train"))
    val_hf = load_from_disk(os.path.join(DATA_CACHE_DIR, "val"))

    train_loader = DataLoader(
        HFImageDataset(train_hf, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        HFImageDataset(val_hf, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with open(HISTORY_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            print(
                f"Epoch {epoch}/{MAX_EPOCHS} - "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            f.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(
                    {"model_state_dict": model.state_dict(), "classes": classes},
                    BEST_CHECKPOINT_PATH,
                )
                print(f"  New best val_loss={val_loss:.4f}, checkpoint saved to {BEST_CHECKPOINT_PATH}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                    break

    print(f"Training finished. Best val_loss={best_val_loss:.4f}. History saved to {HISTORY_CSV_PATH}")


if __name__ == "__main__":
    main()
