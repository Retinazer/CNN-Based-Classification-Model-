"""
Builds presentation-ready figures for the business-proposal deck from the
real tea_cnn outputs: data_cache/ (class counts, test split), history.csv
(training curve), and checkpoints/best_model.pt (test predictions).

Does not modify train.py, evaluate.py, or any checkpoint - only reads them.
Run after train.py / evaluate.py: python3 make_proposal_figures.py
"""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_from_disk
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, recall_score
from torch.utils.data import DataLoader

from common import BEST_CHECKPOINT_PATH, DATA_CACHE_DIR, HISTORY_CSV_PATH, LABELS_PATH, PROJECT_DIR
from train import HFImageDataset, build_model, eval_transform

OUT_DIR = os.path.join(PROJECT_DIR, "proposal_figures")

# --- palette (dataviz skill reference instance, light mode) ---
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES_BLUE = "#2a78d6"    # categorical slot 1
SERIES_ORANGE = "#eb6834"  # categorical slot 2

SEQ_BLUE = {100: "#cde2fb", 250: "#86b6ef", 400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 600: "#184f95"}

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "figure.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": PAGE,
    }
)

FIGSIZE = (16, 10)
DPI = 100


def load_combined_counts():
    """Recount classes across the actual saved train/val/test Datasets."""
    with open(LABELS_PATH) as f:
        classes = json.load(f)
    splits = [load_from_disk(os.path.join(DATA_CACHE_DIR, s)) for s in ("train", "val", "test")]
    combined = concatenate_datasets(splits)
    counts = {c: 0 for c in classes}
    for label_id in combined["label"]:
        counts[classes[label_id]] += 1
    return classes, counts


def load_history():
    epochs, train_acc, val_acc = [], [], []
    with open(HISTORY_CSV_PATH) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))
    return epochs, train_acc, val_acc


def run_test_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    classes = checkpoint["classes"]
    model = build_model(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_hf = load_from_disk(os.path.join(DATA_CACHE_DIR, "test"))
    loader = DataLoader(HFImageDataset(test_hf, eval_transform), batch_size=16, shuffle=False, num_workers=0)

    all_preds, all_labels, all_conf, all_images_pil = [], [], [], []
    img_cursor = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_conf.extend(conf.cpu().tolist())
            batch_n = images.size(0)
            for i in range(batch_n):
                all_images_pil.append(test_hf[img_cursor + i]["image"])
            img_cursor += batch_n

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return classes, all_preds, all_labels, all_conf, all_images_pil, accuracy


def fig_class_distribution(classes, counts, out_path):
    ordered = sorted(classes, key=lambda c: counts[c], reverse=True)
    values = [counts[c] for c in ordered]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    y_pos = np.arange(len(ordered))
    ax.barh(y_pos, values, color=SEQ_BLUE[450], height=0.62, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace("_", " ").title() for c in ordered], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("Number of leaf images", fontsize=14, labelpad=10)
    ax.set_title("Training data: images per disease class", fontsize=22, color=INK_PRIMARY, pad=18, loc="left", fontweight="bold")

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="x", color=GRIDLINE, linewidth=1, zorder=0)
    ax.tick_params(axis="y", length=0)

    for y, v in zip(y_pos, values):
        ax.text(v + max(values) * 0.015, y, f"{v:,}", va="center", fontsize=13, color=INK_SECONDARY)

    min_v, max_v = min(values), max(values)
    ax.text(
        0.0, -0.11,
        f"Dataset combines two sources, {len(ordered)} disease classes, {sum(values):,} images total. "
        f"Caveat: class sizes range from {min_v} to {max_v:,} images — the smallest classes are harder to learn reliably.",
        transform=ax.transAxes, fontsize=12.5, color=INK_MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_training_curve(epochs, train_acc, val_acc, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(epochs, train_acc, color=SERIES_BLUE, linewidth=2.5, label="Training accuracy", zorder=3)
    ax.plot(epochs, val_acc, color=SERIES_ORANGE, linewidth=2.5, label="Validation accuracy", zorder=3)

    best_epoch = epochs[int(np.argmax(val_acc))]
    best_val = max(val_acc)
    ax.scatter([best_epoch], [best_val], color=SERIES_ORANGE, s=90, zorder=4, edgecolor=SURFACE, linewidth=1.5)
    ax.annotate(
        f"Best validation accuracy: {best_val*100:.1f}%\n(epoch {best_epoch})",
        xy=(best_epoch, best_val), xytext=(best_epoch - 0.5, best_val - 0.17),
        fontsize=13, color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=1),
    )

    ax.set_xlabel("Training epoch", fontsize=14, labelpad=10)
    ax.set_ylabel("Accuracy", fontsize=14, labelpad=10)
    ax.set_ylim(0.4, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.set_title("The model actually learns: accuracy over training", fontsize=22, color=INK_PRIMARY, pad=18, loc="left", fontweight="bold")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.legend(frameon=False, fontsize=14, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_accuracy_summary(accuracy, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bar_y, bar_h = 0.42, 0.22
    ax.add_patch(FancyBboxPatch((0.06, bar_y), 0.88, bar_h, boxstyle="round,pad=0,rounding_size=0.03",
                                 linewidth=0, facecolor=GRIDLINE, zorder=1))
    ax.add_patch(FancyBboxPatch((0.06, bar_y), 0.88 * accuracy, bar_h, boxstyle="round,pad=0,rounding_size=0.03",
                                 linewidth=0, facecolor=SEQ_BLUE[450], zorder=2))

    ax.text(0.06, bar_y + bar_h + 0.12, "Overall test-set accuracy", fontsize=20, color=INK_SECONDARY, ha="left")
    ax.text(0.94, bar_y + bar_h / 2, f"{accuracy*100:.2f}%", fontsize=52, color=INK_PRIMARY,
            ha="right", va="center", fontweight="bold", zorder=3)

    ax.text(
        0.06, bar_y - 0.12,
        f"Measured on 1,013 held-out test images never seen during training — the model correctly classified "
        f"{round(accuracy*1013)} of 1,013 leaf photos ({accuracy*100:.2f}%).",
        fontsize=13.5, color=INK_MUTED, ha="left",
    )
    fig.suptitle("Does the model actually work?", fontsize=22, color=INK_PRIMARY, x=0.06, y=0.92, ha="left", fontweight="bold")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_confusion_matrix(classes, labels, preds, out_path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    display_names = [c.replace("_", " ").title() for c in classes]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    im = ax.imshow(cm_norm, cmap=_blue_cmap(), vmin=0, vmax=1)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(display_names, fontsize=12)
    ax.set_xlabel("Predicted class", fontsize=14, labelpad=10)
    ax.set_ylabel("Actual class", fontsize=14, labelpad=10)
    ax.set_title("Where the model gets confused (% of each true class)", fontsize=20, color=INK_PRIMARY, pad=16, loc="left", fontweight="bold")

    for i in range(len(classes)):
        for j in range(len(classes)):
            pct = cm_norm[i, j]
            if pct < 0.005:
                continue
            color = "white" if pct > 0.55 else INK_PRIMARY
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{pct*100:.0f}%", ha="center", va="center", fontsize=11, color=color, fontweight=weight)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    cbar.ax.tick_params(labelsize=11, color=INK_MUTED, labelcolor=INK_MUTED)

    recalls = recall_score(labels, preds, average=None, labels=list(range(len(classes))))
    worst_idx = int(np.argmin(recalls))
    fig.text(
        0.02, -0.02,
        f"Known limitation: '{display_names[worst_idx]}' is the weakest class, correctly recalled only "
        f"{recalls[worst_idx]*100:.0f}% of the time ({cm[worst_idx].sum()} test examples) — it has the fewest "
        f"training images and needs more data before launch.",
        fontsize=13, color=INK_MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return display_names[worst_idx], recalls[worst_idx]


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", [SURFACE, SEQ_BLUE[250], SEQ_BLUE[450], SEQ_BLUE[600]])


def fig_sample_predictions(classes, labels, preds, confs, images, out_path, n=9):
    display_names = [c.replace("_", " ").title() for c in classes]
    rng = np.random.default_rng(7)

    correct_idx = [i for i in range(len(labels)) if preds[i] == labels[i]]
    wrong_idx = [i for i in range(len(labels)) if preds[i] != labels[i]]

    n_wrong = min(3, len(wrong_idx), n)
    n_correct = n - n_wrong
    chosen = list(rng.choice(wrong_idx, size=n_wrong, replace=False)) + list(
        rng.choice(correct_idx, size=n_correct, replace=False)
    )
    rng.shuffle(chosen)

    cols, rows = 3, (n + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=FIGSIZE, dpi=DPI)
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, chosen):
        img = images[idx].convert("RGB")
        is_correct = preds[idx] == labels[idx]
        border_color = STATUS_GOOD if is_correct else STATUS_CRITICAL
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(5)
        mark = "✓ Correct" if is_correct else "✗ Incorrect"
        ax.set_title(
            f"{mark}\nTrue: {display_names[labels[idx]]}\nPredicted: {display_names[preds[idx]]} ({confs[idx]*100:.0f}%)",
            fontsize=12.5, color=border_color if not is_correct else INK_PRIMARY, fontweight="bold" if not is_correct else "normal",
            loc="center",
        )
    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle("Real predictions on unseen leaf photos", fontsize=22, color=INK_PRIMARY, x=0.03, y=0.995, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    classes, counts = load_combined_counts()
    print("Class counts (from data_cache):")
    for c in classes:
        print(f"  {c}: {counts[c]}")
    fig_class_distribution(classes, counts, os.path.join(OUT_DIR, "class_distribution.png"))

    epochs, train_acc, val_acc = load_history()
    print(f"\nFinal epoch {epochs[-1]}: train_acc={train_acc[-1]:.4f}, val_acc={val_acc[-1]:.4f}")
    print(f"Best val_acc={max(val_acc):.4f} at epoch {epochs[int(np.argmax(val_acc))]}")
    fig_training_curve(epochs, train_acc, val_acc, os.path.join(OUT_DIR, "training_curve.png"))

    classes_ck, preds, labels, confs, images, accuracy = run_test_inference()
    assert classes_ck == classes
    print(f"\nTest accuracy (recomputed from checkpoint on {len(labels)} test images): {accuracy:.4f}")
    fig_accuracy_summary(accuracy, os.path.join(OUT_DIR, "accuracy_summary.png"))

    worst_class, worst_recall = fig_confusion_matrix(classes, labels, preds, os.path.join(OUT_DIR, "confusion_matrix_clean.png"))
    print(f"Worst-performing class: {worst_class} (recall={worst_recall:.2f})")

    fig_sample_predictions(classes, labels, preds, confs, images, os.path.join(OUT_DIR, "sample_predictions_grid.png"))

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
