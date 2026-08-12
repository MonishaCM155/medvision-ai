"""
MedVision AI — evaluate a trained checkpoint on a patient-level test split.

Usage:
  python training/evaluate.py --checkpoint training/checkpoints/best.pt \
      --config training/configs/train.yaml
  python training/evaluate.py --checkpoint ... --out artifacts/evaluation

Outputs (machine-readable + human-readable):
  artifacts/evaluation/evaluation_report.json   — all metrics
  artifacts/evaluation/roc_curves.png           — per-class ROC
  artifacts/evaluation/pr_curves.png            — per-class PR
  artifacts/evaluation/calibration_curve.png    — ECE calibration (pooled)
  artifacts/evaluation/threshold_analysis.json  — best-F1 threshold per class

Honesty: if no checkpoint or no dataset is available, the script writes
evaluation_report.json with status "not-executed" and exits 0 — it never
fabricates metrics.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(__file__))

from dataset import (  # noqa: E402
    DEFAULT_LABELS, ChestXrayDataset, get_transform, load_nih_csv, patient_split,
)


# ---------------------------------------------------------------------------
# Metric helpers (no sklearn dependency)
# ---------------------------------------------------------------------------

def binary_metrics(scores, labels, threshold=0.5):
    """Precision, recall, specificity, F1 for one class at `threshold`."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pred = (scores >= threshold).astype(int)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0  # == sensitivity
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "sensitivity": round(recall, 4), "specificity": round(specificity, 4),
            "f1": round(f1, 4), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def auc_from_scores(scores, labels):
    """Binary AUROC via Mann–Whitney U."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    rank_pos = ranks[labels == 1].sum()
    return float((rank_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def pr_curve(scores, labels):
    """Precision/recall sweep → AUPRC (trapezoidal) + curve points."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    s, l = scores[order], labels[order]
    tp = np.cumsum(l)
    fp = np.cumsum(1 - l)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(1.0, float(l.sum()))
    # Start/end anchors
    precision = np.concatenate([[1.0], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [1.0]])
    # Trapezoidal integration (np.trapezoid is numpy>=2; fall back for older)
    try:
        auprc = float(np.trapezoid(precision, recall))
    except AttributeError:
        auprc = float(np.trapz(precision, recall))
    return auprc, precision, recall


def roc_points(scores, labels, n=100):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    ths = np.linspace(scores.min(), scores.max(), n)
    fprs, tprs = [], []
    n_pos = max(1, int(labels.sum()))
    n_neg = max(1, int((1 - labels).sum()))
    for t in ths:
        pred = scores >= t
        tp = float(((pred == 1) & (labels == 1)).sum())
        fp = float(((pred == 1) & (labels == 0)).sum())
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    return fprs, tprs


def ece_pooled(scores, labels, bins=10):
    """Expected calibration error over all pooled (sample, class) sigmoid outputs."""
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, count = 0.0, 0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores >= lo) & (scores < hi) if i < bins - 1 else (scores >= lo)
        if mask.sum() == 0:
            continue
        conf = scores[mask].mean()
        acc = labels[mask].mean()
        ece += mask.sum() * abs(conf - acc)
        count += mask.sum()
    return float(ece / max(1, count))


# ---------------------------------------------------------------------------

def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="MedVision AI — evaluate a chest X-ray checkpoint")
    parser.add_argument("--checkpoint", default=None, help="path to a trained .pt checkpoint")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "train.yaml"))
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", default="artifacts/evaluation")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    ds = cfg.get("dataset", {})
    mod = cfg.get("model", {})
    os.makedirs(args.out, exist_ok=True)

    checkpoint = args.checkpoint
    images_dir = args.images_dir or ds.get("images_dir")
    labels_csv = args.labels_csv or ds.get("labels_csv")
    label_names = list(DEFAULT_LABELS)

    if not checkpoint or not os.path.exists(str(checkpoint)) or not labels_csv or not os.path.exists(str(labels_csv)):
        report = {
            "status": "not-executed",
            "reason": ("Training not executed because the required dataset and/or checkpoint "
                       "are unavailable in the current environment. Evaluation requires a trained "
                       "checkpoint plus NIH ChestX-ray14 images + Data_Entry_2017.csv."),
            "checkpoint": str(checkpoint),
            "labels_csv": str(labels_csv),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(args.out, "evaluation_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("\n[data] Evaluation not executed — no checkpoint and/or dataset available.")
        print("       Run `python training/train.py --config training/configs/train.yaml` first,")
        print("       or use --synthetic-sanity to validate the training pipeline.\n")
        sys.exit(0)

    device = "cpu"
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] Using {'CUDA' if device == 'cuda' else 'MPS' if device == 'mps' else 'CPU'}")

    records, _, skipped = load_nih_csv(str(labels_csv), str(images_dir), label_names, ds.get("label_alias") or {})
    if skipped:
        print(f"[data] skipped {skipped} records with only out-of-vocabulary findings")
    _train, _val, test_records = patient_split(
        records, float(ds.get("train_frac", 0.8)), float(ds.get("val_frac", 0.1)),
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )
    if not test_records:
        print("[data] no test records (all patients in train/val) — nothing to evaluate.")
        sys.exit(0)
    print(f"[data] test records: {len(test_records)}")

    # Load checkpoint
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    num_classes = sd["classifier.weight"].shape[0] if "classifier.weight" in sd else int(mod.get("num_classes", len(label_names)))
    if num_classes != len(label_names):
        label_names = label_names[:num_classes] if num_classes < len(label_names) else label_names
    print(f"[model] checkpoint {checkpoint} · classes={num_classes} · epoch={ckpt.get('epoch')} · kind={ckpt.get('kind')}")

    model = torchvision.models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    test_ds = ChestXrayDataset(
        [r["image_path"] for r in test_records],
        [np.asarray([1.0 if l in r["labels"] else 0.0 for l in label_names], dtype=np.float32) for r in test_records],
        transform=get_transform("val", int(ds.get("input_size", 224))),
    )
    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            out = torch.sigmoid(model(images.to(device))).cpu().numpy()
            all_scores.append(out)
            all_labels.append(targets.numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # Per-class metrics
    per_class = {}
    aucs = []
    for c, lab in enumerate(label_names):
        s, l = scores[:, c], labels[:, c]
        auroc = auc_from_scores(s, l)
        auprc, _, _ = pr_curve(s, l)
        aucs.append(auroc)
        # Best-F1 threshold sweep
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            m = binary_metrics(s, l, threshold=float(t))
            if m["f1"] > best_f1:
                best_t, best_f1 = float(t), m["f1"]
        m50 = binary_metrics(s, l, threshold=0.5)
        per_class[lab] = {
            "auroc": round(auroc, 4) if auroc == auroc else None,
            "auprc": round(auprc, 4),
            "at_0.5": m50,
            "best_f1_threshold": round(best_t, 4),
            "best_f1": round(best_f1, 4),
            "positives": int(l.sum()),
        }

    valid = [per_class[lab]["auroc"] for lab in label_names if per_class[lab]["auroc"] is not None]
    macro_auroc = float(np.mean(valid)) if valid else None
    # Micro AUROC: pool all scores/labels
    micro_auroc = auc_from_scores(scores.ravel(), labels.ravel())
    ece = ece_pooled(scores, labels)

    report = {
        "status": "executed",
        "checkpoint": str(checkpoint),
        "checkpoint_kind": ckpt.get("kind"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "label_names": label_names,
        "test_records": len(test_records),
        "device": device,
        "macro_auroc": round(macro_auroc, 4) if macro_auroc is not None else None,
        "micro_auroc": round(micro_auroc, 4),
        "ece_pooled": round(ece, 4),
        "per_class": per_class,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Threshold analysis JSON
    with open(os.path.join(args.out, "threshold_analysis.json"), "w", encoding="utf-8") as f:
        json.dump({lab: {"best_f1_threshold": per_class[lab]["best_f1_threshold"],
                         "best_f1": per_class[lab]["best_f1"]} for lab in label_names}, f, indent=2)

    # Plots (matplotlib — optional; evaluation still works without it)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        for c, lab in enumerate(label_names):
            if per_class[lab]["auroc"] is None:
                continue
            fprs, tprs = roc_points(scores[:, c], labels[:, c])
            ax.plot(fprs, tprs, lw=1.2, label=f"{lab} ({per_class[lab]['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("Per-class ROC"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "roc_curves.png"), dpi=120); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        for c, lab in enumerate(label_names):
            if per_class[lab]["auroc"] is None:
                continue
            auprc, precision, recall = pr_curve(scores[:, c], labels[:, c])
            ax.plot(recall, precision, lw=1.2, label=f"{lab} ({auprc:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Per-class PR curves"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "pr_curves.png"), dpi=120); plt.close(fig)

        edges = np.linspace(0, 1, 11)
        confs, accs = [], []
        for i in range(10):
            mask = (scores >= edges[i]) & (scores < edges[i + 1])
            if mask.sum():
                confs.append(scores[mask].mean()); accs.append(labels[mask].mean())
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
        ax.plot(confs, accs, "o-", label=f"Model (ECE={ece:.3f})")
        ax.set_xlabel("Mean confidence"); ax.set_ylabel("Observed frequency")
        ax.set_title("Calibration curve (pooled)"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "calibration_curve.png"), dpi=120); plt.close(fig)
    except Exception as exc:
        print(f"[plot] matplotlib unavailable or failed ({exc}) — plots skipped, JSON still written.")

    print(f"\n[result] macro AUROC: {macro_auroc if macro_auroc is not None else 'n/a'} · "
          f"micro AUROC: {micro_auroc:.4f} · ECE: {ece:.4f}")
    for lab in label_names:
        p = per_class[lab]
        print(f"  {lab:<18} AUROC {str(p['auroc']):>7}  AUPRC {p['auprc']:.3f}  "
              f"F1@0.5 {p['at_0.5']['f1']:.3f}  F1@{p['best_f1_threshold']:.2f} {p['best_f1']:.3f}  n+ {p['positives']}")
    print(f"\n[out] saved to {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
