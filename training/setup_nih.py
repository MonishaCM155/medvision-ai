#!/usr/bin/env python
"""
MedVision AI — NIH ChestX-ray14 dataset setup helper.

After downloading NIH images, run this script to verify the layout
and provide instructions for obtaining the labels CSV.

Usage:
    python training/setup_nih.py

Expected directory structure:
    datasets/nih/
    ├── DataEntry_2017.csv      ← labels file (you must download this)
    ├── images/                 ← or IMAGES_DIR.txt pointing to your images
    │   ├── 00000001_000.png
    │   └── ...
    └── IMAGES_DIR.txt          ← optional: path to external image directory
"""

import os
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIH_DIR = os.path.join(ROOT, "datasets", "nih")


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("=" * 60)
    print("MedVision AI — NIH ChestX-ray14 Dataset Setup")
    print("=" * 60)

    # Check for CSV
    csv_path = os.path.join(NIH_DIR, "DataEntry_2017.csv")
    has_csv = os.path.exists(csv_path)

    # Check for images
    img_dir = os.path.join(NIH_DIR, "images")
    dir_marker = os.path.join(NIH_DIR, "IMAGES_DIR.txt")
    external_dir = None
    if os.path.exists(dir_marker):
        with open(dir_marker, encoding="utf-8") as f:
            external_dir = f.read().strip()

    actual_img_dir = img_dir
    if not os.path.isdir(img_dir) and external_dir and os.path.isdir(external_dir):
        actual_img_dir = external_dir

    # Count images
    n_images = 0
    if os.path.isdir(actual_img_dir):
        for root, dirs, files in os.walk(actual_img_dir):
            n_images += sum(1 for f in files if f.lower().endswith(".png"))

    print(f"\n[1] Labels CSV: {'FOUND' if has_csv else 'MISSING'}")
    if has_csv:
        # Count CSV rows
        with open(csv_path, encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1  # minus header
        print(f"    {csv_path}")
        print(f"    {rows:,} annotation rows")
    else:
        print(f"    Expected at: {csv_path}")
        print()
        print("    DOWNLOAD OPTIONS:")
        print("    +-------------------------------------------------------+")
        print("    | Option A - Kaggle (easiest, free account required):   |")
        print("    |   https://www.kaggle.com/datasets/nih-chest-xrays/data|")
        print("    |   Download 'Data Entry 2017.csv' -> place at:         |")
        print(f"    |   {csv_path}")
        print("    |                                                       |")
        print("    | Option B - NIH Box (no account needed):               |")
        print("    |   https://nihcc.app.box.com/v/ChestXray-NIHCC         |")
        print("    |   Download 'Data Entry 2017.csv' -> place at same path|")
        print("    +-------------------------------------------------------+")

    print(f"\n[2] Images: {'FOUND' if n_images > 0 else 'MISSING'}")
    if n_images > 0:
        print(f"    Directory: {actual_img_dir}")
        print(f"    {n_images:,} PNG files found")
    else:
        print(f"    Expected at: {img_dir}")
        if external_dir:
            print(f"    External dir: {external_dir} (not found)")

    print(f"\n[3] Status:")
    if has_csv and n_images > 0:
        print("    Dataset is READY for training!")
        print()
        print("    Next steps:")
        print("      python training/ingest_multi.py")
        print("      python training/train.py --config training/configs/train.yaml")
    elif has_csv:
        print("    WARNING: CSV found but images missing.")
        print("    Place images in datasets/nih/images/")
    elif n_images > 0:
        print("    WARNING: Images found but CSV missing.")
        print("    Download DataEntry_2017.csv from the sources above.")
    else:
        print("    Neither CSV nor images found.")
        print("    Download NIH ChestX-ray14 from one of the sources above.")

    print()
    return 0 if (has_csv and n_images > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
