"""
MedVision AI — per-class decision thresholds selected on the VALIDATION set.

Rule (STEP 8 of the integration spec): operating thresholds are chosen on the
validation partition ONLY, after training, before any test-set evaluation.
The test set is never used to tune thresholds.

Selection objective: maximize per-class F1 over a [0.05, 0.95] sweep.
Classes with no validation positives keep the global default 0.5 (there is no
signal to tune on — this is reported, not hidden).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import ChestXrayDataset, get_transform  # noqa: E402


@torch.no_grad()
def validation_scores(model, records, label_names, input_size=224, batch_size=32, device="cpu"):
    """Sigmoid scores + labels for the given (validation) records."""
    ds = ChestXrayDataset(
        [r["image_path"] for r in records],
        [np.asarray([1.0 if l in r["labels"] else 0.0 for l in label_names], dtype=np.float32) for r in records],
        transform=get_transform("val", input_size),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_scores, all_labels = [], []
    for images, targets in loader:
        out = torch.sigmoid(model(images.to(device))).cpu().numpy()
        all_scores.append(out)
        all_labels.append(targets.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def _f1_at(scores, labels, threshold):
    pred = (scores >= threshold).astype(int)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def select_thresholds(model, records, label_names, input_size=224, batch_size=32,
                      device="cpu", default=0.5, sweep=None, min_positives=1):
    """Per-class F1-maximizing thresholds on validation records.

    Returns (thresholds dict, meta dict). `meta` records per-class validation
    F1 at the chosen threshold and notes classes kept at default because they
    lacked validation positives.
    """
    sweep = sweep if sweep is not None else np.linspace(0.05, 0.95, 19)
    scores, labels = validation_scores(model, records, label_names, input_size, batch_size, device)
    thresholds, meta, f1s = {}, {}, {}
    for c, lab in enumerate(label_names):
        s, l = scores[:, c], labels[:, c]
        n_pos = int(l.sum())
        best_t, best_f1 = default, _f1_at(s, l, default)
        if n_pos >= min_positives:
            for t in sweep:
                f1 = _f1_at(s, l, float(t))
                if f1 > best_f1:
                    best_t, best_f1 = float(t), f1
        thresholds[lab] = best_t
        f1s[lab] = round(best_f1, 4)
        meta[lab] = {
            "threshold": best_t,
            "val_f1_at_threshold": round(best_f1, 4),
            "val_positives": int(n_pos),
            "tuned": n_pos >= min_positives,
        }
    return thresholds, {"per_class": meta, "policy": "validate-max-f1",
                        "note": "Thresholds maximize per-class F1 on the validation partition only. "
                                "The test set is never used to tune thresholds."}
