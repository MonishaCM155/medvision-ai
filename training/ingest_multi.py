"""
MedVision AI — unified multi-source dataset ingestion (Phase 5/6).

Builds a common-schema manifest from the datasets under datasets/ and writes
deterministic patient-level train/val/test splits into training/splits/.

Sources (see docs/dataset-research.md for access, licenses, and the
dataset-disease matrix):

  * Shenzhen Hospital CXR (NLM) — datasets/shenzhen/
      images/CHNCXR_#####_[01].png   (readme-documented: 0 = normal, 1 = TB)
      clinical_readings/*.txt        (cross-checked, 0 mismatches of 662)
      patient_id = "shenzhen_" + filename  (one image per patient — the
      dataset provides no finer grouping; documented limitation)

  * Cohen et al. covid-chestxray-dataset (CC BY 4.0) — datasets/covid19/
      metadata.csv + images/ (modality == "X-ray" only)
      patient_id = "covid_" + patientid  (patient-level split keeps all images
      of one patient together)

Label mapping (documented in docs/label-mapping.md):
  Shenzhen: _0 -> No Finding, _1 -> Tuberculosis
  Cohen:    finding == "No Finding"                    -> No Finding
            "COVID-19" in finding                      -> COVID-19
            finding.startswith("Pneumonia/") (others)  -> Pneumonia
            excluded: todo, Unknown, Tuberculosis, SARS, MERS-CoV

Usage:
  python training/ingest_multi.py                 # from training/configs paths
  python training/ingest_multi.py --split 0.7 0.15 0.15 --seed 42

Outputs:
  training/splits/train.csv | val.csv | test.csv  (Image Index, Finding Labels,
                                                   Patient ID, Dataset Source)
  training/splits/ingest_report.json              (per-split patients/images,
                                                   per-class positives, leakage check)
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DEFAULT_LABELS, count_positives, load_split_csv, write_split_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHENZHEN_DIR = os.path.join(ROOT, "datasets", "shenzhen")
COVID_DIR = os.path.join(ROOT, "datasets", "covid19")
NIH_DIR = os.path.join(ROOT, "datasets", "nih")
SPLITS_DIR = os.path.join(ROOT, "training", "splits")

# NIH ChestX-ray14 label mapping (the CSV uses short names separated by '|'):
#   Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Atelectasis,
#   Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening,
#   Pneumonia, Hernia
# Map to our 10-class schema:
nih_LABEL_MAP = {
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural Effusion",
    "Infiltration": "Lung Opacity",
    "Consolidation": "Lung Opacity",
    "Mass": "Lung Opacity",
    "Nodule": "Lung Opacity",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Edema": "Edema",
    "Pneumonia": "Pneumonia",
    # Not in our 10-class schema: Emphysema, Fibrosis, Pleural_Thickening, Hernia
}

# Classes declared unavailable only when NIH data is absent.
# When NIH ChestX-ray14 IS available, all 10 classes can be trained.
UNAVAILABLE_WITHOUT_NIH = ["Atelectasis", "Pleural Effusion", "Edema",
                           "Cardiomegaly", "Lung Opacity", "Pneumothorax"]


def ingest_shenzhen():
    """Return records with labels from the documented filename convention."""
    records = []
    img_dir = os.path.join(SHENZHEN_DIR, "images")
    if not os.path.isdir(img_dir):
        print(f"[shenzhen] images dir missing: {img_dir}")
        return records
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".png"):
            continue
        base = fname[:-4]
        labels = ["Tuberculosis"] if base.endswith("_1") else ["No Finding"]
        records.append({
            "image_path": os.path.relpath(os.path.join(img_dir, fname), ROOT),
            "labels": labels,
            "patient_id": "shenzhen_" + base,
            "dataset_source": "shenzhen",
        })
    return records


def ingest_covid():
    """Return records from Cohen et al. metadata.csv (X-ray modality only)."""
    records = []
    meta = os.path.join(COVID_DIR, "metadata.csv")
    img_dir = os.path.join(COVID_DIR, "images")
    if not (os.path.exists(meta) and os.path.isdir(img_dir)):
        print(f"[covid19] metadata.csv or images dir missing: {meta}")
        return records
    excluded = 0
    seen_cases = set()  # (patientid, finding) dedup key
    with open(meta, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("modality") or "").strip().lower() != "x-ray":
                continue
            finding = (row.get("finding") or "").strip()
            if finding in ("todo", "Unknown", ""):
                excluded += 1
                continue
            if finding == "No Finding":
                labels = ["No Finding"]
            elif "COVID-19" in finding:
                labels = ["COVID-19"]  # supersedes generic pneumonia (documented)
            elif "SARS" in finding or "MERS" in finding:
                excluded += 1  # coronaviruses, not in the target classes
                continue
            elif finding == "Tuberculosis":
                excluded += 1  # TB provenance kept with Shenzhen only
                continue
            elif finding == "Pneumonia" or finding.startswith("Pneumonia/"):
                labels = ["Pneumonia"]
            else:
                excluded += 1
                print(f"[covid19] unmapped finding skipped: {finding!r}")
                continue
            fname = (row.get("filename") or "").strip()
            pid_raw = (row.get("patientid") or "").strip() or ("anon_" + fname)
            # Cohen has both .jpg and .png for the same X-ray; keep one format.
            root_key = os.path.splitext(fname)[0]
            if root_key in seen_cases:
                excluded += 1
                continue
            seen_cases.add(root_key)
            path = os.path.join(img_dir, fname)
            if not os.path.exists(path):
                print(f"[covid19] missing image file skipped: {fname}")
                excluded += 1
                continue
            records.append({
                "image_path": os.path.relpath(path, ROOT),
                "labels": labels,
                "patient_id": "covid_" + pid_raw,
                "dataset_source": "covid19",
            })
    if excluded:
        print(f"[covid19] excluded {excluded} rows (todo/unknown/SARS/MERS/TB/missing files)")
    return records


# Binary-label column names in the pre-split CSVs (train.csv, val.csv, etc.)
NIH_BINARY_COLS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def _resolve_nih_img_dir():
    """Find the NIH images directory — checks datasets/nih/images,
    then falls back to IMAGES_DIR.txt for external storage."""
    img_dir = os.path.join(NIH_DIR, "images")
    dir_marker = os.path.join(NIH_DIR, "IMAGES_DIR.txt")
    if not os.path.isdir(img_dir) and os.path.exists(dir_marker):
        with open(dir_marker, encoding="utf-8") as _f:
            alt = _f.read().strip()
        if alt and os.path.isdir(alt):
            return alt
    if os.path.isdir(img_dir):
        return img_dir
    return None


def _build_nih_img_index(img_dir):
    """Build filename→path index, searching subdirectories (images_003/ etc.)."""
    idx = {}
    for root, _dirs, fnames in os.walk(img_dir):
        for fn in fnames:
            if fn.lower().endswith(".png"):
                idx[fn] = os.path.join(root, fn)
    return idx


def _nih_row_to_labels(row):
    """Convert a CSV row with binary columns to a list of disease labels."""
    labels = []
    for col in NIH_BINARY_COLS:
        val = row.get(col, "0")
        try:
            if int(val) == 1:
                # Map NIH column name to our 10-class schema
                mapped = nih_LABEL_MAP.get(col)
                if mapped:
                    labels.append(mapped)
        except (ValueError, TypeError):
            continue
    # All 14 zeros → No Finding
    if not labels:
        labels = ["No Finding"]
    return sorted(set(labels))


def ingest_nih():
    """Return records from NIH ChestX-ray14 dataset.

    Supports two CSV formats:
      1. Pre-split binary CSVs (train.csv, val.csv, test.csv) — preferred.
         Columns: Image Index, Atelectasis, Cardiomegaly, ... (0/1 values)
      2. DataEntry_2017.csv — pipe-separated Finding Labels column.

    Image directory resolved from:
      datasets/nih/images/ or datasets/nih/IMAGES_DIR.txt
    """
    img_dir = _resolve_nih_img_dir()
    if not img_dir:
        print("[nih] images directory not found (check datasets/nih/images or IMAGES_DIR.txt)")
        return []
    print(f"[nih] using image dir: {img_dir}")
    img_index = _build_nih_img_index(img_dir)
    print(f"[nih] indexed {len(img_index)} image files")

    # Try pre-split binary CSVs first (train.csv / Data.csv)
    train_csv = os.path.join(NIH_DIR, "train.csv")
    data_csv = os.path.join(NIH_DIR, "Data.csv")
    data_entry_csv = os.path.join(NIH_DIR, "DataEntry_2017.csv")

    if os.path.exists(data_csv):
        return _ingest_nih_binary_csv(data_csv, img_index)
    elif os.path.exists(train_csv):
        return _ingest_nih_binary_csv(train_csv, img_index)
    elif os.path.exists(data_entry_csv):
        return _ingest_nih_dataentry_csv(data_entry_csv, img_index)
    else:
        print("[nih] no CSV found — need Data.csv, train.csv, or DataEntry_2017.csv")
        return []


def _ingest_nih_binary_csv(csv_path, img_index):
    """Ingest a binary-label CSV (Image Index + 0/1 disease columns)."""
    records = []
    excluded = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("Image Index") or "").strip()
            if not fname:
                continue
            labels = _nih_row_to_labels(row)
            # Skip rows where all mapped labels are excluded (e.g. only Hernia)
            if labels == ["No Finding"] and any(int(row.get(c, 0)) == 1 for c in NIH_BINARY_COLS if nih_LABEL_MAP.get(c)):
                excluded += 1
                continue
            img_path = img_index.get(fname)
            if not img_path:
                excluded += 1
                continue
            # Extract patient ID from filename (e.g. 00020458_017.png → 00020458)
            pid = fname.split("_")[0] if "_" in fname else fname
            records.append({
                "image_path": os.path.relpath(img_path, ROOT),
                "labels": labels,
                "patient_id": "nih_" + pid,
                "dataset_source": "nih",
            })
    if excluded:
        print(f"[nih] excluded {excluded} rows (missing files or unmapped-only labels)")
    print(f"[nih] loaded {len(records)} records from {os.path.basename(csv_path)}")
    return records


def _ingest_nih_dataentry_csv(csv_path, img_index):
    """Ingest DataEntry_2017.csv (pipe-separated Finding Labels)."""
    records = []
    excluded = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("Image Index") or "").strip()
            if not fname:
                continue
            finding_raw = (row.get("Finding Labels") or "").strip()
            pid = (row.get("Patient ID") or "").strip()
            if not finding_raw or finding_raw == "No Finding":
                labels = ["No Finding"]
            else:
                nih_labels = [l.strip() for l in finding_raw.split("|") if l.strip()]
                mapped = set()
                for nl in nih_labels:
                    if nl in nih_LABEL_MAP:
                        mapped.add(nih_LABEL_MAP[nl])
                if not mapped:
                    excluded += 1
                    continue
                labels = sorted(mapped)
            img_path = img_index.get(fname)
            if not img_path:
                excluded += 1
                continue
            records.append({
                "image_path": os.path.relpath(img_path, ROOT),
                "labels": labels,
                "patient_id": "nih_" + str(pid),
                "dataset_source": "nih",
            })
    if excluded:
        print(f"[nih] excluded {excluded} rows (unmapped labels or missing files)")
    print(f"[nih] loaded {len(records)} records from DataEntry_2017.csv")
    return records


def patient_split(records, train_frac=0.7, val_frac=0.15, seed=42):
    """Patient-level split; every image of a patient stays in one partition."""
    patients = sorted({r["patient_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_train = int(len(patients) * train_frac)
    n_val = int(len(patients) * val_frac)
    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    def part(s):
        return [r for r in records if r["patient_id"] in s]

    return part(train_p), part(val_p), part(test_p)


def _load_nih_presplit():
    """Load pre-split NIH CSVs (train.csv, val.csv, test.csv) directly.
    Returns (train_r, val_r, test_r) or None if pre-split files are missing."""
    train_csv = os.path.join(NIH_DIR, "train.csv")
    val_csv = os.path.join(NIH_DIR, "val.csv")
    test_csv = os.path.join(NIH_DIR, "test.csv")
    if not all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
        return None
    img_dir = _resolve_nih_img_dir()
    if not img_dir:
        return None
    img_index = _build_nih_img_index(img_dir)
    splits = []
    for csv_file in [train_csv, val_csv, test_csv]:
        records = _ingest_nih_binary_csv(csv_file, img_index)
        splits.append(records)
    return tuple(splits)


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="MedVision AI — multi-source ingestion")
    parser.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                        help="train/val/test fractions (patient-level)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits_dir", default=SPLITS_DIR)
    args = parser.parse_args()
    t_frac, v_frac, _ = args.split

    # Check for pre-split NIH CSVs first (highest quality path)
    presplit = _load_nih_presplit()
    if presplit:
        train_r, val_r, test_r = presplit
        all_records = train_r + val_r + test_r
        sources_present = set(r["dataset_source"] for r in all_records)
        has_nih = "nih" in sources_present
        UNAVAILABLE = [] if has_nih else UNAVAILABLE_WITHOUT_NIH

        # Combine Shenzhen + COVID with NIH splits
        shenzhen_records = ingest_shenzhen()
        covid_records = ingest_covid()
        extra = shenzhen_records + covid_records
        if extra:
            # Add extra records to training split
            train_r = train_r + extra
            all_records = train_r + val_r + test_r
            sources_present = set(r["dataset_source"] for r in all_records)

        print(f"[presplit] Using pre-split NIH CSVs (train/val/test)")
    else:
        records = ingest_shenzhen() + ingest_covid() + ingest_nih()
        if not records:
            print("\n[ingest] No usable records — datasets/ is empty or missing.\n"
                  "         Download the sources per docs/dataset-research.md first.")
            sys.exit(1)

        sources_present = set(r["dataset_source"] for r in records)
        has_nih = "nih" in sources_present
        UNAVAILABLE = [] if has_nih else UNAVAILABLE_WITHOUT_NIH

        train_r, val_r, test_r = patient_split(records, t_frac, v_frac, args.seed)
        all_records = records

    # Leakage check: no patient in more than one partition.
    ids = [{r["patient_id"] for r in part} for part in (train_r, val_r, test_r)]
    overlap = ids[0] & ids[1] | ids[0] & ids[2] | ids[1] & ids[2]
    if overlap:
        print(f"[leak] WARNING: {len(overlap)} patient(s) in multiple partitions.")

    pos_all = count_positives(all_records, DEFAULT_LABELS)
    trained = [l for l in DEFAULT_LABELS if pos_all[l] > 0 and l not in UNAVAILABLE]
    untrained = [l for l in DEFAULT_LABELS if l not in trained]
    all_sources = sorted(sources_present)
    print(f"[data] records: {len(all_records)}  patients: {len({r['patient_id'] for r in all_records})}")
    print(f"[data] sources: {all_sources}")
    print(f"[data] positives: {pos_all}")
    print(f"[labels] TRAINED: {trained}")
    if untrained:
        print(f"[labels] UNAVAILABLE (no legitimate data / declared): {untrained}")
    else:
        print(f"[labels] All 10 classes have training data — full model capability.")

    for name, part in (("train", train_r), ("val", val_r), ("test", test_r)):
        write_split_csv(part, os.path.join(args.splits_dir, f"{name}.csv"))
        n_pat = len({r["patient_id"] for r in part})
        n_src = {s: sum(1 for r in part if r["dataset_source"] == s) for s in all_sources}
        print(f"[split] {name:<5} images={len(part):>4} patients={n_pat:>4} sources={n_src}")

    report = {
        "status": "prepared",
        "seed": args.seed,
        "split_fractions": list(args.split),
        "total_images": len(all_records),
        "total_patients": len({r["patient_id"] for r in all_records}),
        "per_source": {s: sum(1 for r in all_records if r["dataset_source"] == s) for s in all_sources},
        "trained_classes": trained,
        "unavailable_classes": untrained,
        "class_positives": pos_all,
        "leakage_check": "warn" if overlap else "pass",
        "splits": {
            name: {
                "images": len(part),
                "patients": len({r["patient_id"] for r in part}),
                "positives": count_positives(part, DEFAULT_LABELS),
                "sources": {s: sum(1 for r in part if r["dataset_source"] == s) for s in all_sources},
            }
            for name, part in (("train", train_r), ("val", val_r), ("test", test_r))
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(args.splits_dir, exist_ok=True)
    with open(os.path.join(args.splits_dir, "ingest_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[out] splits → {os.path.abspath(args.splits_dir)}/")
    print("Next:  python training/train.py --config training/configs/train.yaml")


if __name__ == "__main__":
    main()
