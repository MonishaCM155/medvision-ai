"""
PyTorch dataset tooling for MedVision AI chest X-ray training.

Supports the NIH ChestX-ray14 layout (Data_Entry_2017.csv with Image Index /
Finding Labels / Patient ID) with:

  * multi-label one-hot encoding (10 thoracic pathology labels)
  * patient-level train/val/test splits (never image-level — prevents leakage)
  * training augmentation (RandomResizedCrop / HorizontalFlip / ColorJitter)
  * inverse-frequency positive weights for BCEWithLogitsLoss

CheXpert/MIMIC-CXR require their own converters; the loaders here are the
documented entry point for NIH-format data.
"""

import csv
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

DEFAULT_LABELS = [
    "No Finding", "Pneumonia", "COVID-19", "Tuberculosis", "Cardiomegaly",
    "Pleural Effusion", "Edema", "Atelectasis", "Pneumothorax", "Lung Opacity",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ChestXrayDataset(Dataset):
    """Multi-label chest X-ray dataset from (image_paths, labels) arrays."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = list(image_paths)
        self.labels = [np.asarray(l, dtype=np.float32) for l in labels]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


def load_nih_csv(labels_csv: str, images_dir: str, label_names: List[str], label_alias: dict = None):
    """Parse NIH ChestX-ray14 Data_Entry_2017.csv into usable records.

    Returns (records, label_names):
      records — list of {"image_path", "labels", "patient_id"}
      label_names — the label list actually used (unchanged input)

    Records whose findings contain ONLY labels outside `label_names` (e.g.
    "Hernia") are skipped so they never teach the model a wrong "No Finding".
    """
    alias = label_alias or {}
    records = []
    skipped = 0
    known = set(label_names)
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = (row.get("Image Index") or "").strip()
            if not img:
                continue
            raw = [l.strip() for l in (row.get("Finding Labels") or "No Finding").split("|") if l.strip()]
            mapped = {alias.get(l, l) for l in raw}
            if "No Finding" in mapped:
                labels = ["No Finding"]
            else:
                labels = sorted(mapped & known)
            if not labels:
                skipped += 1
                continue  # only unknown findings — cannot represent without lying
            records.append({
                "image_path": os.path.join(images_dir, img),
                "labels": labels,
                "patient_id": (row.get("Patient ID") or f"anon_{len(records)}").strip(),
            })
    return records, label_names, skipped


def labels_to_vector(record_labels: List[str], label_names: List[str]) -> np.ndarray:
    v = np.zeros(len(label_names), dtype=np.float32)
    if "No Finding" in record_labels:
        return v
    for lab in record_labels:
        if lab in label_names:
            v[label_names.index(lab)] = 1.0
    return v


def patient_split(records: list, train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42):
    """Patient-level split — every image of a patient stays in one partition."""
    patients = sorted({r["patient_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_train = int(len(patients) * train_frac)
    n_val = int(len(patients) * val_frac)
    train_p, val_p = set(patients[:n_train]), set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    def partition(patient_set):
        return [r for r in records if r["patient_id"] in patient_set]

    return partition(train_p), partition(val_p), partition(test_p)


def get_transform(mode: str = "train", input_size: int = 224):
    if mode == "train":
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def compute_pos_weight(records: list, label_names: List[str], alpha: float = 1.0) -> torch.Tensor:
    """Inverse-frequency positive weights for BCEWithLogitsLoss (per class)."""
    n = max(1, len(records))
    pos = np.zeros(len(label_names), dtype=np.float64)
    for r in records:
        pos += labels_to_vector(r["labels"], label_names)
    neg = n - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        pw = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    if alpha <= 0:
        pw = np.ones_like(pw)
    else:
        pw = pw * alpha
    return torch.tensor(np.clip(pw, 0.1, 50.0), dtype=torch.float32)


def count_positives(records: list, label_names: List[str]) -> dict:
    pos = np.zeros(len(label_names), dtype=int)
    for r in records:
        pos += labels_to_vector(r["labels"], label_names).astype(int)
    return {lab: int(cnt) for lab, cnt in zip(label_names, pos)}


def make_loaders(records_train, records_val, label_names, batch_size, input_size=224, num_workers=0):
    """Build DataLoaders from split record lists."""
    train_ds = ChestXrayDataset(
        [r["image_path"] for r in records_train],
        [labels_to_vector(r["labels"], label_names) for r in records_train],
        transform=get_transform("train", input_size),
    )
    val_ds = ChestXrayDataset(
        [r["image_path"] for r in records_val],
        [labels_to_vector(r["labels"], label_names) for r in records_val],
        transform=get_transform("val", input_size),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Synthetic sanity dataset — validates the full pipeline without real data.
# Images carry a weak class-correlated intensity signature so the pipeline can
# be exercised end-to-end (loss moves, AUROC computed) without pretending to
# be a real medical dataset.
# ---------------------------------------------------------------------------

def make_synthetic_dataset(out_dir: str, n_patients: int = 24, images_per_patient: int = 3,
                           size: int = 256, seed: int = 7) -> Tuple[str, List[dict]]:
    """Generate a tiny synthetic CXR-like dataset + labels CSV; returns (csv, records)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    csv_rows = []
    for p in range(n_patients):
        pid = f"SYN-{p:03d}"
        for k in range(images_per_patient):
            img = rng.normal(100, 12, (size, size)).astype(np.float32)
            # Weak class signature: Pneumonia raises right-lower intensity.
            if p % 3 == 0:
                img[int(size * 0.6):int(size * 0.85), int(size * 0.6):int(size * 0.9)] += 25
                labels = "Pneumonia"
            elif p % 3 == 1:
                img[int(size * 0.1):int(size * 0.35), int(size * 0.35):int(size * 0.65)] += 20
                labels = "Cardiomegaly"
            else:
                labels = "No Finding"
            img = np.clip(img, 0, 255).astype(np.uint8)
            fname = f"syn_{pid}_{k}.png"
            Image.fromarray(img, "L").save(os.path.join(out_dir, fname))
            records.append({"image_path": os.path.join(out_dir, fname), "labels": labels.split("|"), "patient_id": pid})
            csv_rows.append({"Image Index": fname, "Finding Labels": labels, "Patient ID": pid})
    csv_path = os.path.join(out_dir, "synthetic_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Image Index", "Finding Labels", "Patient ID"])
        writer.writeheader()
        writer.writerows(csv_rows)
    return csv_path, records
