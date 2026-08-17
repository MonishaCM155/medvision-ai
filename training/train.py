"""
MedVision AI — DenseNet-121 multi-label chest X-ray training.

Usage:
  python training/train.py --config training/configs/train.yaml
  python training/train.py --config training/configs/train.yaml --epochs 20 --batch_size 32
  python training/train.py --config ... --synthetic-sanity     # validate the pipeline
      end-to-end on a tiny synthetic dataset (no real data required)

Features:
  * Device auto-detection: CUDA → MPS → CPU (never requires a GPU)
  * Patient-level train/val split (no leakage)
  * BCEWithLogitsLoss with inverse-frequency positive weighting
  * Adam/SGD + cosine/step LR scheduling + optional mixed precision (CUDA)
  * Checkpointing (best by val macro-AUROC + last), resume, early stopping
  * Experiment metadata JSON (config, metrics, device, timestamps)

If the dataset files are missing and --synthetic-sanity is not given, the
script reports honestly and exits without pretending training occurred.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(__file__))

from dataset import (  # noqa: E402
    DEFAULT_LABELS, ChestXrayDataset, available_classes, compute_pos_weight,
    count_positives, get_transform, load_nih_csv, load_split_csv,
    make_synthetic_dataset, patient_split,
)
from thresholds import select_thresholds  # noqa: E402


# ---------------------------------------------------------------------------
# AUROC without sklearn (Mann–Whitney U, rank-based)
# ---------------------------------------------------------------------------

def auc_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary AUROC. scores/labels are 1-D tensors."""
    scores = scores.detach().cpu().float()
    labels = labels.detach().cpu().float()
    n_pos = int(labels.sum())
    n_neg = int(labels.numel()) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    rank_pos = ranks[labels == 1].sum()
    auc = (rank_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def macro_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean binary AUROC across classes; NaN classes ignored."""
    vals = [auc_from_scores(scores[:, c], labels[:, c]) for c in range(labels.shape[1])]
    vals = [v for v in vals if v == v]  # drop NaN
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(args):
    cfg = {}
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        print(f"[config] WARNING: config file {args.config} not found — using defaults + CLI flags.")

    # CLI overrides (only when explicitly provided)
    overrides = {
        "images_dir": args.images_dir, "labels_csv": args.labels_csv,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "checkpoint_dir": args.checkpoint_dir, "resume": args.resume,
        "device": args.device,
    }
    for key, val in overrides.items():
        if val is not None:
            if key in ("epochs", "batch_size", "seed"):
                cfg.setdefault("training", {})[key] = int(val)
            elif key in ("lr",):
                cfg.setdefault("training", {})[key] = float(val)
            elif key == "device":
                cfg["device"] = str(val)
            else:
                cfg.setdefault("training", {})[key] = str(val)
    return cfg


def resolve_device(choice: str) -> str:
    if choice not in ("auto", ""):
        if choice not in ("cuda", "mps", "cpu"):
            print(f"[device] Unknown device '{choice}' — falling back to auto.")
            choice = "auto"
        else:
            return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(num_classes: int, pretrained: bool, device: str):
    """DenseNet-121 with a PLAIN Linear multi-label head.

    The head must be a single Linear layer: the inference engine
    (backend/app/main.py) detects a fine-tuned checkpoint via the
    'classifier.weight' key and swaps the head accordingly.
    """
    weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    model.to(device)
    return model


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_amp_context(device: str, enabled: bool):
    """AMP autocast context manager compatible with torch >= 2.0."""
    if enabled and device == "cuda":
        try:
            return torch.amp.autocast("cuda")
        except (TypeError, AttributeError):
            from torch.cuda.amp import autocast  # legacy

            return autocast()
    return _NullContext()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, amp=True):
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with make_amp_context(device, amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    losses, n = 0.0, 0
    all_scores, all_labels = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with make_amp_context(device, amp):
            outputs = model(images)
        losses += nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction="sum").item()
        n += targets.size(0)
        all_scores.append(outputs.cpu())
        all_labels.append(targets.cpu())
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    val_loss = losses / max(1, n)
    return val_loss, macro_auc(torch.sigmoid(scores), labels)


def save_checkpoint(path, model, optimizer, epoch, val_auc, cfg, kind, metadata=None):
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "val_macro_auc": val_auc,
        "config": cfg,
        "kind": kind,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)
    print(f"[checkpoint] saved {kind} → {path}")


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="MedVision AI — train DenseNet-121 chest X-ray classifier")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "train.yaml"))
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--synthetic-sanity", action="store_true",
                        help="run 2 epochs on a tiny synthetic dataset to validate the pipeline")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    args = parser.parse_args()

    cfg = load_config(args)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    print(f"[device] Using {'CUDA' if device == 'cuda' else 'MPS' if device == 'mps' else 'CPU'}")

    exp = cfg.get("experiment", {})
    ds = cfg.get("dataset", {})
    mod = cfg.get("model", {})
    tr = cfg.get("training", {})
    label_names = list(DEFAULT_LABELS)
    num_classes = int(mod.get("num_classes", len(label_names)))
    if num_classes != len(label_names):
        print(f"[config] WARNING: num_classes={num_classes} but the engine expects a 10 (or 14) class head for real inference.")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    images_dir = (args.synthetic_sanity and args.images_dir) or ds.get("images_dir")
    labels_csv = (args.synthetic_sanity and args.labels_csv) or ds.get("labels_csv")
    splits_dir = os.path.abspath(str(ds.get("splits_dir") or "training/splits"))

    # Reproducible patient-level splits on disk (multi-source ingest or NIH
    # prepare_dataset.py) are the canonical data source. NIH-format CSV
    # derivation is the fallback; synthetic sanity validates the loop.
    train_records = load_split_csv(os.path.join(splits_dir, "train.csv"))
    val_records = load_split_csv(os.path.join(splits_dir, "val.csv"))
    _test_records = load_split_csv(os.path.join(splits_dir, "test.csv"))

    if args.synthetic_sanity:
        sanity_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "sanity")
        os.makedirs(sanity_dir, exist_ok=True)
        csv_path, records = make_synthetic_dataset(os.path.join(sanity_dir, "data"))
        labels_csv, images_dir = csv_path, os.path.dirname(csv_path)
        ckpt_dir = os.path.join(sanity_dir, "runs")
        epochs = int(tr.get("epochs", 50))
        if args.epochs is None:
            epochs = 2  # sanity: 2 epochs is enough to prove the loop
        print("[data] SYNTHETIC SANITY MODE — tiny synthetic dataset, pipeline validation only.")
        label_alias = {}
        train_records, val_records, _test_records = patient_split(
            records, float(ds.get("train_frac", 0.8)), float(ds.get("val_frac", 0.1)), seed=seed)
    elif train_records:
        records = train_records + val_records + _test_records
        ckpt_dir = str(tr.get("checkpoint_dir") or "training/checkpoints")
        epochs = int(tr.get("epochs", 50))
        label_alias = ds.get("label_alias") or {}
        print(f"[data] loaded reproducible patient-level splits from {splits_dir}")
    elif images_dir and labels_csv and os.path.exists(str(labels_csv)):
        # Fallback: derive patient-level splits directly from an NIH-format CSV.
        records, _, skipped = load_nih_csv(str(labels_csv), str(images_dir), label_names, ds.get("label_alias") or {})
        ckpt_dir = str(tr.get("checkpoint_dir") or "training/checkpoints")
        epochs = int(tr.get("epochs", 50))
        label_alias = ds.get("label_alias") or {}
        if skipped:
            print(f"[data] skipped {skipped} records with only out-of-vocabulary findings")
        train_records, val_records, _test_records = patient_split(
            records, float(ds.get("train_frac", 0.8)), float(ds.get("val_frac", 0.1)), seed=seed)
        print(f"[data] splits derived from CSV (patient-level, seed={seed})")
    else:
        print("\n[data] Training not executed because the required dataset is unavailable in the current environment.")
        print("       Run  python training/ingest_multi.py  after obtaining the datasets per docs/dataset-research.md,")
        print("       or run with --synthetic-sanity to validate the pipeline on a tiny synthetic dataset.\n")
        sys.exit(0)

    print(f"[data] records: train={len(train_records)} val={len(val_records)}")
    print(f"[data] positives: {count_positives(records, label_names)}")

    unavailable = list(ds.get("unavailable_classes") or [])
    trained_classes, untrained_classes = available_classes(records, label_names, unavailable)
    print(f"[labels] trained: {', '.join(trained_classes)}")
    if untrained_classes:
        print(f"[labels] unavailable (no training signal — kept at 0, never reported as findings): {', '.join(untrained_classes)}")

    batch_size = int(args.batch_size if args.batch_size else tr.get("batch_size", 32))
    input_size = int(ds.get("input_size", 224))

    def to_vectors(records):
        return [np.asarray([1.0 if l in r["labels"] else 0.0 for l in label_names], dtype=np.float32) for r in records]

    train_loader = torch.utils.data.DataLoader(
        ChestXrayDataset([r["image_path"] for r in train_records], to_vectors(train_records),
                         transform=get_transform("train", input_size)),
        batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        ChestXrayDataset([r["image_path"] for r in val_records], to_vectors(val_records),
                         transform=get_transform("val", input_size)),
        batch_size=batch_size, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Model / loss / optimizer / scheduler
    # ------------------------------------------------------------------
    model = build_model(num_classes, bool(mod.get("pretrained", True)), device)
    # Class weights from the TRAINING partition only (val/test never inform the loss).
    pw = compute_pos_weight(train_records, label_names, alpha=float(tr.get("pos_weight_alpha", 1.0)))
    print(f"[loss] pos_weight = {[round(float(v), 2) for v in pw]}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))

    lr = float(args.lr if args.lr else tr.get("lr", 1e-4))
    wd = float(tr.get("weight_decay", 1e-4))
    if str(tr.get("optimizer", "adam")).lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=float(tr.get("momentum", 0.9)), weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler_name = str(tr.get("scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(tr.get("step_size", 15)), gamma=float(tr.get("gamma", 0.1)))
    else:
        scheduler = None

    scaler = None
    amp = not args.no_amp
    if device == "cuda" and amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (TypeError, AttributeError):
            from torch.cuda.amp import GradScaler  # legacy

            scaler = GradScaler()

    start_epoch = 0
    best_auc = float("-inf")
    patience = int(tr.get("early_stopping_patience", 8))
    stale = 0
    resume_path = args.resume or tr.get("resume")
    if resume_path and os.path.exists(str(resume_path)):
        ckpt = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if ckpt.get("optimizer") and not args.synthetic_sanity:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_auc = float(ckpt.get("val_macro_auc", float("-inf")))
        print(f"[resume] resumed from {resume_path} at epoch {start_epoch} (best AUC {best_auc:.4f})")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.synthetic_sanity:
        # Sanity artifacts stay inside the sanity dir — training/results/ must
        # only ever contain artifacts from a real dataset run.
        results_dir = os.path.join(os.path.dirname(ckpt_dir), "results")
    else:
        results_dir = str(tr.get("results_dir") or "training/results")
    os.makedirs(results_dir, exist_ok=True)
    ckpt_meta = {
        "label_names": list(label_names),
        "label_alias": label_alias,
        "unavailable_classes": list(untrained_classes),
        "trained_classes": list(trained_classes),
        "dataset": {"labels_csv": str(labels_csv), "images_dir": str(images_dir),
                    "splits_dir": str(splits_dir)},
        "threshold_policy": str(tr.get("threshold_policy", "validate-max-f1")),
        "seed": seed,
    }
    run_meta = {
        "experiment": exp.get("name", "densenet121-chestxray"),
        "seed": seed, "device": device, "dataset": str(labels_csv),
        "label_names": label_names, "num_classes": num_classes,
        "trained_classes": trained_classes, "unavailable_classes": untrained_classes,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "optimizer": str(tr.get("optimizer", "adam")), "scheduler": scheduler_name,
        "pos_weight": [round(float(v), 3) for v in pw],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "epochs_log": [],
    }
    print(f"[train] epochs={epochs} batch_size={batch_size} lr={lr} device={device} "
          f"scheduler={scheduler_name} amp={amp and device == 'cuda'}")
    print(f"[train] checkpoints → {ckpt_dir}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp)
        val_loss, val_auc = evaluate(model, val_loader, device, amp)
        lr_now = optimizer.param_groups[0]["lr"]
        log_entry = {"epoch": epoch, "train_loss": round(train_loss, 4),
                     "val_loss": round(val_loss, 4), "val_macro_auc": round(val_auc, 4),
                     "lr": lr_now, "seconds": round(time.time() - t0, 1)}
        run_meta["epochs_log"].append(log_entry)
        print(f"[epoch {epoch:02d}/{epochs - 1:02d}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_macro_auc={val_auc:.4f} lr={lr_now:.2e} "
              f"({log_entry['seconds']}s)")

        improved = val_auc == val_auc and val_auc > best_auc
        if improved:
            best_auc = val_auc
            stale = 0
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optimizer, epoch, val_auc, cfg, "best", ckpt_meta)
        else:
            stale += 1
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer, epoch, val_auc, cfg, "last", ckpt_meta)
        with open(os.path.join(ckpt_dir, "experiment.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

        if scheduler is not None:
            scheduler.step()
        if stale >= patience and epoch > 5:
            print(f"[early-stop] no improvement for {patience} epochs — stopping.")
            break

    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_meta["best_val_macro_auc"] = round(best_auc, 4) if best_auc != float("-inf") else None
    with open(os.path.join(ckpt_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    # ------------------------------------------------------------------
    # Per-class decision thresholds — validation set ONLY (never the test set)
    # ------------------------------------------------------------------
    thresholds = None
    if str(tr.get("threshold_policy", "validate-max-f1")) == "validate-max-f1" and val_records:
        print("[thresholds] selecting per-class F1-max thresholds on the VALIDATION set only ...")
        thresholds, thr_meta = select_thresholds(model, val_records, label_names,
                                                 input_size=input_size, batch_size=batch_size, device=device)
        print(f"[thresholds] {json.dumps(thresholds)}")
        for lab in untrained_classes:
            thr_meta["per_class"][lab]["note"] = "unavailable class — threshold is inert (never reported)"
        # Freeze thresholds into both checkpoints so the engine uses them.
        for fn in ("best.pt", "last.pt"):
            p = os.path.join(ckpt_dir, fn)
            if os.path.exists(p):
                c = torch.load(p, map_location="cpu")
                c.setdefault("metadata", {}).update({"thresholds": thresholds, "threshold_meta": thr_meta})
                torch.save(c, p)
        with open(os.path.join(results_dir, "thresholds.json"), "w", encoding="utf-8") as f:
            json.dump({"thresholds": thresholds, "meta": thr_meta}, f, indent=2)
    else:
        print("[thresholds] keeping global 0.5 per class (threshold_policy != validate-max-f1)")

    # ------------------------------------------------------------------
    # training/results/ artifacts (machine-readable)
    # ------------------------------------------------------------------
    with open(os.path.join(results_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta["epochs_log"], f, indent=2)
    with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "status": "executed",
            "experiment": run_meta["experiment"], "device": device,
            "best_val_macro_auc": run_meta["best_val_macro_auc"],
            "threshold_policy": str(tr.get("threshold_policy", "validate-max-f1")),
            "thresholds": thresholds,
            "started_at": run_meta["started_at"], "finished_at": run_meta["finished_at"],
            "note": "Validation metrics only — test-set metrics are produced by "
                    "training/evaluate.py on the held-out test partition.",
        }, f, indent=2)
    try:
        import yaml
        with open(os.path.join(results_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception as exc:
        print(f"[results] config_used.yaml skipped ({exc})")
    # Validation classification report (labeled clearly — NOT a test report).
    if thresholds is not None:
        from dataset import labels_to_vector
        vec = np.asarray([labels_to_vector(r["labels"], label_names) for r in val_records])
        val_report = {"partition": "validation",
                      "note": "Class statistics at frozen thresholds on the validation partition. "
                              "Test-set classification report: run training/evaluate.py.",
                      "per_class": {}}
        for c, lab in enumerate(label_names):
            val_report["per_class"][lab] = {"frozen_threshold": thresholds[lab],
                                             "val_positives": int(vec[:, c].sum()),
                                             "val_negatives": int(len(val_records) - vec[:, c].sum())}
        with open(os.path.join(results_dir, "val_classification_report.json"), "w", encoding="utf-8") as f:
            json.dump(val_report, f, indent=2)

    print(f"[done] best val macro-AUROC: {best_auc if best_auc != float('-inf') else 'n/a'} → {ckpt_dir}")
    print(f"[done] artifacts → {os.path.abspath(results_dir)}/")


if __name__ == "__main__":
    main()
