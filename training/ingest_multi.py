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
SPLITS_DIR = os.path.join(ROOT, "training", "splits")

# Classes declared unavailable for this data mix (never trained, never reported).
UNAVAILABLE = ["Atelectasis", "Pleural Effusion", "Edema", "Cardiomegaly",
               "Lung Opacity", "Pneumothorax"]


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

    records = ingest_shenzhen() + ingest_covid()
    if not records:
        print("\n[ingest] No usable records — datasets/ is empty or missing.\n"
              "         Download the sources per docs/dataset-research.md first.")
        sys.exit(1)

    train_r, val_r, test_r = patient_split(records, t_frac, v_frac, args.seed)

    # Leakage check: no patient in more than one partition.
    ids = [{r["patient_id"] for r in part} for part in (train_r, val_r, test_r)]
    overlap = ids[0] & ids[1] | ids[0] & ids[2] | ids[1] & ids[2]
    if overlap:
        print(f"[leak] FATAL: {len(overlap)} patient(s) in multiple partitions — aborting.")
        sys.exit(1)

    pos_all = count_positives(records, DEFAULT_LABELS)
    trained = [l for l in DEFAULT_LABELS if pos_all[l] > 0 and l not in UNAVAILABLE]
    untrained = [l for l in DEFAULT_LABELS if l not in trained]
    print(f"[data] records: {len(records)}  patients: {len({r['patient_id'] for r in records})}")
    print(f"[data] per-source: { {s: sum(1 for r in records if r['dataset_source'] == s) for s in ('shenzhen', 'covid19')} }")
    print(f"[data] positives: {pos_all}")
    print(f"[labels] TRAINED: {trained}")
    print(f"[labels] UNAVAILABLE (no legitimate data / declared): {untrained}")

    for name, part in (("train", train_r), ("val", val_r), ("test", test_r)):
        write_split_csv(part, os.path.join(args.splits_dir, f"{name}.csv"))
        n_pat = len({r["patient_id"] for r in part})
        n_src = {s: sum(1 for r in part if r["dataset_source"] == s) for s in ("shenzhen", "covid19")}
        print(f"[split] {name:<5} images={len(part):>4} patients={n_pat:>4} sources={n_src}")

    report = {
        "status": "prepared",
        "seed": args.seed,
        "split_fractions": list(args.split),
        "total_images": len(records),
        "total_patients": len({r["patient_id"] for r in records}),
        "per_source": {s: sum(1 for r in records if r["dataset_source"] == s) for s in ("shenzhen", "covid19")},
        "trained_classes": trained,
        "unavailable_classes": untrained,
        "class_positives": pos_all,
        "leakage_check": "pass",
        "splits": {
            name: {
                "images": len(part),
                "patients": len({r["patient_id"] for r in part}),
                "positives": count_positives(part, DEFAULT_LABELS),
                "sources": {s: sum(1 for r in part if r["dataset_source"] == s) for s in ("shenzhen", "covid19")},
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
