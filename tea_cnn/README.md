# Tea Leaf Disease Classifier (CNN, transfer learning)

Combines two Hugging Face datasets — `yunusserhat/tea_sickness_dataset` and
`Project-AgML/tea_leaf_disease_classification` — into one canonical 9-class
label space (their overlapping classes are merged; see `common.py`'s
`CANONICAL_LABEL_MAP`), then trains a MobileNetV2 transfer-learning
classifier on the combined data.

## How to run

Run in order from inside `tea_cnn/`:

```bash
pip3 install --break-system-packages -r requirements.txt
python3 load_data.py   # downloads + merges both datasets, writes data_cache/{train,val,test}
python3 train.py       # trains MobileNetV2 head, early stopping, writes checkpoints/best_model.pt + history.csv
python3 evaluate.py    # evaluates best checkpoint on the test split
```

If you hit `SSL: CERTIFICATE_VERIFY_FAILED` downloading pretrained weights
(common with python.org's macOS Python build), run:

```bash
pip3 install --break-system-packages certifi
export SSL_CERT_FILE=$(python3 -m certifi)
```

before `python3 train.py`.

## Output files

- `data_cache/{train,val,test}/` — the stratified 70/15/15 split of the combined dataset (Hugging Face `Dataset` saved to disk).
- `data_cache/labels.json` — the 9 canonical class names, in label-index order.
- `samples/sample_<class>.jpg` — one example image per canonical class, saved by `load_data.py` for a visual sanity check.
- `checkpoints/best_model.pt` — best model weights (lowest validation loss) plus the class list, saved by `train.py`.
- `history.csv` — per-epoch train/val loss and accuracy, written by `train.py`.
- `confusion_matrix.png` — test-set confusion matrix across all 9 classes, written by `evaluate.py`.
- `sample_predictions.png` — grid of 12 test images with true label, predicted label, and confidence, written by `evaluate.py`.

## Results from the actual run

**Combined dataset:** 6,752 images total after merging and label reconciliation.
Split: train=4,726, val=1,013, test=1,013 (stratified).

Class counts in the combined dataset:

| class | count |
|---|---|
| algal_spot | 1113 |
| anthracnose | 100 |
| bird_eye_spot | 100 |
| brown_blight | 980 |
| gray_blight | 1100 |
| healthy | 1074 |
| helopeltis | 1000 |
| red_spot | 1143 |
| white_spot | 142 |

**Training:** MobileNetV2 (ImageNet-pretrained backbone, replaced classifier head),
early-stopped at epoch 18/30 (patience=5). Best checkpoint at epoch 13:
val_loss=0.0878, val_acc=0.9654.

**Test set evaluation:** overall accuracy = **0.9635** (975/1013).

Per-class precision/recall/F1 (test set):

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| algal_spot | 0.99 | 0.98 | 0.98 | 167 |
| anthracnose | 0.78 | 0.47 | 0.58 | 15 |
| bird_eye_spot | 0.74 | 0.93 | 0.82 | 15 |
| brown_blight | 1.00 | 0.92 | 0.96 | 147 |
| gray_blight | 0.92 | 0.99 | 0.95 | 165 |
| healthy | 0.95 | 1.00 | 0.97 | 161 |
| helopeltis | 1.00 | 0.97 | 0.99 | 150 |
| red_spot | 0.99 | 0.97 | 0.98 | 172 |
| white_spot | 0.80 | 0.95 | 0.87 | 21 |

**Known limitation:** `anthracnose` has both the lowest recall (0.47) and one
of the smallest test supports (15 examples) — it comes only from the smaller
`yunusserhat` dataset (100 total examples), so the model sees relatively few
examples of it. `bird_eye_spot` and `white_spot` are similarly small-sample
classes (15 and 21 test examples) with lower precision than the large classes.
More data for these three classes would likely be the highest-value next step.
