"""
MedVision AI — NIH ChestX-ray14 dataset preparation and reproducible splitting.

This script does NOT download or fabricate data. It validates a dataset that
you have already obtained, reports honest per-class statistics, and writes the
patient-level train/val/test splits that training, threshold selection, and
evaluation all consume.

Usage:
  python training/prepare_dataset.py                          # paths from training/configs/train.yaml
  python training/prepare_dataset.py --labels_csv /path/Data_Entry_2017.csv \\
                                     --images_dir /path/images
  python training/prepare_dataset.py --check-images          # verify every CSV row has a file

Outputs:
  training/splits/train.csv | val.csv | test.csv   — (Image Index, Finding Labels, Patient ID)
  training/splits/split_report.json                — patients/images per split, per-class
                                                     positives/negatives/imbalance, mapping table,
                                                     unavailable classes (machine-readable)

NIH ChestX-ray14 acquisition (required before this script does anything):
  1. Apply for access at  https://nihcc.app.box.com/v/ChestXray-NIHCC
     (public research download; no clinical credentials required).
  2. Download the 12 image batches (images_001.zip … images_012.zip, ~42 GB)
     and extract them into one images/ directory (the PNG files are flat).
  3. Download Data_Entry_2017.csv from the same location and place it next to
     images/.
  4. Run this script.

Honesty contract: this script never invents images, labels, or patients. If
the CSV or images are missing it reports exactly what is missing and exits
without writing any splits.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (  # noqa: E402
    DEFAULT_LABELS, available_classes, count_positives, load_nih_csv,
    patient_split, write_split_csv,
)

# Transparent mapping table (single source of truth for documentation output).
MAPPING_TABLE = [
    ("No Finding", "No Finding", "literal NIH label"),
    ("Pneumonia", "Pneumonia", "literal NIH label"),
    ("Cardiomegaly", "Cardiomegaly", "literal NIH label"),
    ("Pleural Effusion", "Effusion", "NIH synonym"),
    ("Edema", "Edema", "literal NIH label"),
    ("Atelectasis", "Atelectasis", "literal NIH label"),
    ("Pneumothorax", "Pneumothorax", "literal NIH label"),
    ("Lung Opacity", "Consolidation, Infiltration", "CheXpert-style merge (opacity = consolidation|infiltration)"),
    ("COVID-19", "—", "UNAVAILABLE — NIH ChestXray14 (2017) has no COVID-19 cases"),
    ("Tuberculosis", "—", "UNAVAILABLE — TB is a separate NIH collection, not ChestXray14"),
]


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="MedVision AI — prepare NIH ChestX-ray14 splits")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "train.yaml"))
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--splits_dir", default=None)
    parser.add_argument("--check-images", action="store_true",
                        help="verify every CSV row resolves to an existing image file (slow: ~112k files)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    ds = cfg.get("dataset", {})
    labels_csv = args.labels_csv or ds.get("labels_csv")
    images_dir = args.images_dir or ds.get("images_dir")
    splits_dir = args.splits_dir or ds.get("splits_dir", "training/splits")
    seed = args.seed if args.seed is not None else int(cfg.get("experiment", {}).get("seed", 42))
    unavailable = list(ds.get("unavailable_classes") or [])

    missing = [name for name, p in (("labels_csv", labels_csv), ("images_dir", images_dir)) if not p or not os.path.exists(str(p))]
    if missing:
        print("\n[data] Dataset preparation not executed — the following are missing in this environment:")
        for name in missing:
            print(f"         - {name}: {locals().get(name)}")
        print("\n       You must provide, from the official NIH ChestX-ray14 download "
              "(https://nihcc.app.box.com/v/ChestXray-NIHCC):")
        print("         1. Data_Entry_2017.csv")
        print("         2. the images/ directory (extracted from images_001.zip … images_012.zip)\n")
        sys.exit(0)

    records, label_names, skipped = load_nih_csv(str(labels_csv), str(images_dir), DEFAULT_LABELS, ds.get("label_alias") or {})
    # Persist split CSVs with root-relative image paths (portable across checkouts).
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for r in records:
        r["image_path"] = os.path.relpath(r["image_path"], root)
    print(f"[data] parsed {len(records)} records from {os.path.basename(str(labels_csv))} "
          f"(skipped {skipped} with only out-of-vocabulary findings)")

    if not records:
        print("[data] No usable records — aborting without writing splits.")
        sys.exit(1)

    if args.check_images:
        missing_files = [r["image_path"] for r in records if not os.path.exists(r["image_path"])]
        print(f"[check] image existence: {len(records) - len(missing_files)}/{len(records)} present"
              + (f" · {len(missing_files)} MISSING (e.g. {missing_files[0]})" if missing_files else ""))
        if missing_files:
            print("[check] ABORT — the images directory does not match Data_Entry_2017.csv. "
                  "Verify extraction (files are flat PNGs) before preparing splits.")
            sys.exit(1)

    trained, untrained = available_classes(records, label_names, unavailable)
    pos = count_positives(records, label_names)
    print(f"\n[labels] trained classes ({len(trained)}): {', '.join(trained)}")
    if untrained:
        print(f"[labels] NOT trained / unavailable ({len(untrained)}): {', '.join(untrained)}"
              + (" — zero positives in this dataset" if any(pos[l] == 0 for l in untrained) else ""))

    train_records, val_records, test_records = patient_split(
        records, float(ds.get("train_frac", 0.8)), float(ds.get("val_frac", 0.1)), seed=seed)

    def _stats(recs):
        return {
            "patients": len({r["patient_id"] for r in recs}),
            "images": len(recs),
            "positives": count_positives(recs, label_names),
            "negatives": {lab: len(recs) - count_positives(recs, label_names)[lab] for lab in label_names},
        }

    report = {
        "status": "prepared",
        "dataset": "NIH ChestXray14",
        "labels_csv": str(labels_csv),
        "images_dir": str(images_dir),
        "seed": seed,
        "mapping": [{"project_label": a, "nih_labels": b, "note": c} for a, b, c in MAPPING_TABLE],
        "unavailable_classes": untrained,
        "trained_classes": trained,
        "label_names": label_names,
        "splits": {
            "train": _stats(train_records),
            "val": _stats(val_records),
            "test": _stats(test_records),
        },
        "imbalance_ratio": {lab: round(pos[lab] / max(1, len(records) - pos[lab]), 4) for lab in label_names},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(splits_dir, exist_ok=True)
    write_split_csv(train_records, os.path.join(splits_dir, "train.csv"))
    write_split_csv(val_records, os.path.join(splits_dir, "val.csv"))
    write_split_csv(test_records, os.path.join(splits_dir, "test.csv"))
    with open(os.path.join(splits_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[splits] patient-level splits written (no patient appears in more than one partition):")
    for name in ("train", "val", "test"):
        s = report["splits"][name]
        print(f"  {name:<6} patients={s['patients']:>6}  images={s['images']:>7}")
    print("\n[out] " + os.path.abspath(splits_dir) + "/")
    print("\nNext:  python training/train.py --config training/configs/train.yaml\n")


if __name__ == "__main__":
    main()
