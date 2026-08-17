"""
MedVision AI — select per-class decision thresholds on the VALIDATION set and
freeze them into a trained checkpoint (STEP 8 of the integration spec).

Usage:
  python training/select_thresholds.py --checkpoint training/checkpoints/best.pt \\
      --config training/configs/train.yaml
  python training/select_thresholds.py --checkpoint ... --out training/results/thresholds.json

Rules:
  * The validation partition is the ONLY data used to pick thresholds.
  * The test partition is never touched here.
  * Objective: maximize per-class F1 over a [0.05, 0.95] sweep.
  * Thresholds are written into the checkpoint metadata (metadata.thresholds)
    so the inference engine applies them automatically, plus a JSON copy.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DEFAULT_LABELS, load_nih_csv, load_split_csv, patient_split  # noqa: E402
from thresholds import select_thresholds  # noqa: E402


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="MedVision AI — validation-set threshold selection")
    parser.add_argument("--checkpoint", required=True, help="path to a trained .pt checkpoint")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "train.yaml"))
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--out", default="training/results/thresholds.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--no-patch-checkpoint", action="store_true",
                        help="only write the JSON; do not modify the checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"\n[data] Threshold selection not executed — checkpoint not found: {args.checkpoint}\n")
        sys.exit(0)

    cfg = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    ds = cfg.get("dataset", {})
    labels_csv = args.labels_csv or ds.get("labels_csv")
    images_dir = args.images_dir or ds.get("images_dir")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("metadata") or {}
    label_names = list(meta.get("label_names") or DEFAULT_LABELS)
    num_classes = ckpt["state_dict"]["classifier.weight"].shape[0]
    if num_classes < len(label_names):
        label_names = label_names[:num_classes]
    print(f"[model] checkpoint {args.checkpoint} · classes={num_classes} · labels={label_names}")

    # Validation records: persisted splits preferred, else derive from CSV.
    splits_dir = str(ds.get("splits_dir") or "training/splits")
    val_records = load_split_csv(os.path.join(splits_dir, "val.csv"), str(images_dir))
    if not val_records:
        if not labels_csv or not os.path.exists(str(labels_csv)):
            print(f"\n[data] Threshold selection not executed — no validation split and no labels CSV available.\n")
            sys.exit(0)
        records, _, _ = load_nih_csv(str(labels_csv), str(images_dir), label_names, ds.get("label_alias") or {})
        _, val_records, _ = patient_split(records, float(ds.get("train_frac", 0.8)),
                                          float(ds.get("val_frac", 0.1)),
                                          seed=int(cfg.get("experiment", {}).get("seed", 42)))
    print(f"[data] validation records: {len(val_records)}")

    device = "cpu"
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] Using {'CUDA' if device == 'cuda' else 'MPS' if device == 'mps' else 'CPU'}")

    import torch.nn as nn
    import torchvision
    model = torchvision.models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()})
    model.to(device)

    thresholds, thr_meta = select_thresholds(model, val_records, label_names,
                                             input_size=int(ds.get("input_size", 224)),
                                             batch_size=args.batch_size, device=device)
    print(f"[thresholds] {json.dumps(thresholds, indent=2)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"thresholds": thresholds, "meta": thr_meta,
                   "checkpoint": args.checkpoint,
                   "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()},
                  f, indent=2)
    print(f"[out] {os.path.abspath(args.out)}")

    if not args.no_patch_checkpoint:
        ckpt.setdefault("metadata", {}).update({"thresholds": thresholds, "threshold_meta": thr_meta})
        torch.save(ckpt, args.checkpoint)
        print(f"[checkpoint] thresholds frozen into {args.checkpoint} (metadata.thresholds) — "
              "the engine will apply them automatically.")


if __name__ == "__main__":
    main()
