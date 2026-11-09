"""
Loads the best checkpoint from train.py and evaluates it on the held-out
test split: overall accuracy, per-class precision/recall/F1, a confusion
matrix plot, and a grid of sample predictions.

Run after train.py: python3 evaluate.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from common import BEST_CHECKPOINT_PATH, DATA_CACHE_DIR, PROJECT_DIR
from train import HFImageDataset, build_model, eval_transform

CONFUSION_MATRIX_PATH = os.path.join(PROJECT_DIR, "confusion_matrix.png")
SAMPLE_PREDICTIONS_PATH = os.path.join(PROJECT_DIR, "sample_predictions.png")
NUM_SAMPLE_PREDICTIONS = 12

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def unnormalize(tensor):
    return (tensor * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    classes = checkpoint["classes"]
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    model = build_model(num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_hf = load_from_disk(os.path.join(DATA_CACHE_DIR, "test"))
    test_loader = DataLoader(
        HFImageDataset(test_hf, eval_transform), batch_size=16, shuffle=False, num_workers=0
    )

    all_preds, all_labels, all_confidences, all_images = [], [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_images.extend(images.cpu())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\nOverall test accuracy: {accuracy:.4f}")

    print("\nPer-class precision/recall/F1:")
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")

    n = min(NUM_SAMPLE_PREDICTIONS, len(all_images))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(all_images), size=n, replace=False)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for ax_i, idx in zip(axes, sample_idx):
        img = unnormalize(all_images[idx]).permute(1, 2, 0).numpy()
        true_label = classes[all_labels[idx]]
        pred_label = classes[all_preds[idx]]
        confidence = all_confidences[idx]
        ax_i.imshow(img)
        ax_i.axis("off")
        color = "green" if true_label == pred_label else "red"
        ax_i.set_title(f"true: {true_label}\npred: {pred_label} ({confidence:.2f})", fontsize=8, color=color)
    for ax_i in axes[n:]:
        ax_i.axis("off")
    plt.tight_layout()
    plt.savefig(SAMPLE_PREDICTIONS_PATH)
    plt.close(fig)
    print(f"Saved sample predictions grid to {SAMPLE_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
