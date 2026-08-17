"""
MedVision AI — pre-scale dataset images for faster training (originals kept).

Loads every image referenced by training/splits/*.csv, converts to RGB,
downscales the longest side to --max_side (default 1024, Lanczos), and saves a
JPEG (quality 90) under datasets/preprocessed/<source>/<basename>.jpg. The
split CSVs are rewritten to point at the preprocessed copies.

This does NOT change the model input resolution (the training transform still
resizes to 224) — it only removes the repeated full-resolution decode cost.
Original downloads remain untouched under datasets/.

Usage:
  python training/preprocess_images.py              # pre-scale + rewrite splits
  python training/preprocess_images.py --dry-run    # report only, no writes
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PROJECT_ROOT, load_split_csv, write_split_csv  # noqa: E402

SPLIT_DIR = os.path.join(PROJECT_ROOT, "training", "splits")
OUT_ROOT = os.path.join(PROJECT_ROOT, "datasets", "preprocessed")


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="MedVision AI — pre-scale split images")
    parser.add_argument("--max_side", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from PIL import Image

    total = 0
    for split in ("train", "val", "test"):
        path = os.path.join(SPLIT_DIR, f"{split}.csv")
        records = load_split_csv(path)
        if not records:
            print(f"[{split}] no records — skip")
            continue
        new_records = []
        t0 = time.time()
        for r in records:
            src = r["image_path"]
            if not os.path.exists(src):
                print(f"  [skip] missing {src}")
                new_records.append(r)
                continue
            # Source from the Dataset Source column (ingest_multi) or inferred
            # from the path when absent (NIH fallback: datasets/<name>/...).
            source = (r.get("dataset_source") or "").strip()
            if not source:
                rel = src.replace("\\", "/")
                parts = rel.split("/")
                source = parts[1] if len(parts) > 3 else "unknown"
            base = os.path.basename(src)
            out_rel = f"datasets/preprocessed/{source}/{base.rsplit('.', 1)[0]}.jpg"
            out = os.path.join(PROJECT_ROOT, out_rel)
            if not args.dry_run:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                im = Image.open(src).convert("RGB")
                if max(im.size) > args.max_side:
                    scale = args.max_side / float(max(im.size))
                    im = im.resize((max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))),
                                   Image.LANCZOS)
                im.save(out, "JPEG", quality=90)
            r["image_path"] = out_rel
            new_records.append(r)
            total += 1
        if not args.dry_run:
            write_split_csv(new_records, path)
        print(f"[{split}] {len(new_records)} images ({time.time() - t0:.1f}s){' [dry-run]' if args.dry_run else ''}")

    print(f"\n[out] total {total} images -> {OUT_ROOT.replace(os.sep, '/')}/"
          f"{'  (dry-run — no writes)' if args.dry_run else '  splits rewritten'}")


if __name__ == "__main__":
    main()
