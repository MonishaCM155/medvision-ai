"""
pytest tests for the engine's fine-tuned checkpoint integration.

These tests build a synthetic DenseNet-121 checkpoint (random-initialized, 10
classes) with the metadata the training pipeline writes (class order, frozen
validation-selected thresholds, unavailable classes, dataset, epoch, val AUROC)
and verify the engine:

  * prefers best.pt and loads a 10/14-class head as fine-tuned
  * exposes honest provenance through _engine_meta (checkpointFile, dataset,
    trainedClasses, unavailableClasses, thresholds, valMacroAuc)
  * applies the frozen (validation-selected) thresholds in predictions
  * marks unavailable classes trained:false and never selects them as the top
    diagnosis
  * reports backbone-live mode (no provenance fields) when no checkpoint exists

Run: pytest tests/test_checkpoint_integration.py -v
"""

import base64
import io
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
torchvision = pytest.importorskip("torchvision", reason="torchvision not installed")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.main import (  # noqa: E402
    DISEASE_LABELS, _build_engine, _build_prediction, _class_thresholds, _engine_meta,
)

LABELS = list(DISEASE_LABELS)
UNAVAILABLE = ["COVID-19", "Tuberculosis"]
THRESHOLDS = {lab: 0.5 for lab in LABELS}
THRESHOLDS.update({"Pneumonia": 0.62, "Cardiomegaly": 0.31, "No Finding": 0.72})


def _fake_checkpoint(path, thresholds=None):
    model = torchvision.models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(LABELS))
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "epoch": 31,
        "kind": "best",
        "val_macro_auc": 0.8123,
        "timestamp": "2026-01-15T10:00:00+00:00",
        "config": {"dataset": {"label_alias": {"Effusion": "Pleural Effusion"}}},
        "metadata": {
            "label_names": LABELS,
            "trained_classes": [l for l in LABELS if l not in UNAVAILABLE],
            "unavailable_classes": UNAVAILABLE,
            "thresholds": thresholds or THRESHOLDS,
            "threshold_policy": "validate-max-f1",
            "dataset": {"labels_csv": "Data_Entry_2017.csv", "images_dir": "images"},
            "seed": 42,
        },
    }
    torch.save(payload, path)


def _synthetic_cxr_data_url(size=512, seed=0):
    from PIL import Image, ImageDraw

    img = Image.new("L", (size, size), 90)
    d = ImageDraw.Draw(img)
    d.rectangle([int(size * 0.40), int(size * 0.08), int(size * 0.60), int(size * 0.92)], fill=175)
    d.rectangle([int(size * 0.25), int(size * 0.05), int(size * 0.75), int(size * 0.32)], fill=135)
    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.default_rng(seed).normal(0, 6, arr.shape).astype(np.float32)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture()
def fine_tuned_engine(tmp_path):
    """An engine built against a temp dir containing a fake best.pt."""
    os.environ["MEDVISION_SKIP_PRETRAINED"] = "1"
    _fake_checkpoint(str(tmp_path / "best.pt"))
    return _build_engine(str(tmp_path))


def test_engine_loads_best_pt_as_fine_tuned(fine_tuned_engine):
    assert fine_tuned_engine is not None, "engine must build even with a fake checkpoint"
    assert fine_tuned_engine["num_classes"] == 10
    assert fine_tuned_engine["checkpoint"] is not None
    assert os.path.basename(fine_tuned_engine["checkpoint"]) == "best.pt"


def test_engine_meta_provenance(fine_tuned_engine):
    meta = _engine_meta(fine_tuned_engine)
    assert meta["engineMode"] == "real-model"
    assert meta["source"] == "pytorch-checkpoint"
    assert meta["predictionSource"] == "real-inference"
    assert meta["weightsLoaded"] is True
    assert meta["checkpointFile"] == "best.pt"
    assert meta["checkpointKind"] == "best"
    assert meta["checkpointEpoch"] == 31
    assert meta["checkpointValMacroAuc"] == 0.8123
    assert meta["trainingDate"] == "2026-01-15T10:00:00+00:00"
    assert meta["dataset"] == "Data_Entry_2017.csv"
    assert set(meta["unavailableClasses"]) == set(UNAVAILABLE)
    assert set(meta["trainedClasses"]) == set(LABELS) - set(UNAVAILABLE)
    assert meta["thresholds"]["Pneumonia"] == 0.62
    assert meta["thresholdPolicy"] == "validate-max-f1"


def test_class_thresholds_from_checkpoint(fine_tuned_engine):
    thr = _class_thresholds(fine_tuned_engine)
    assert thr["Pneumonia"] == 0.62
    assert thr["Cardiomegaly"] == 0.31
    assert thr["No Finding"] == 0.72
    # classes without a frozen threshold keep the 0.5 default
    assert thr["Atelectasis"] == 0.5


def test_predictions_flag_unavailable_classes_and_exclude_top(fine_tuned_engine):
    pred = _build_prediction("cxr.png", _synthetic_cxr_data_url(), "DenseNet121",
                             False, False, fine_tuned_engine)
    by_name = {d["disease"]: d for d in pred["diseases"]}
    assert len(by_name) == 10
    # unavailable classes are flagged, never presented as findings
    assert by_name["COVID-19"]["trained"] is False
    assert by_name["Tuberculosis"]["trained"] is False
    assert "note" in by_name["COVID-19"]
    assert by_name["Pneumonia"]["trained"] is True
    # top diagnosis must be a trained class
    assert pred["topDiagnosis"] not in UNAVAILABLE
    # frozen thresholds are applied in the output
    assert by_name["Pneumonia"]["threshold"] == 0.62
    assert by_name["No Finding"]["threshold"] == 0.72
    # provenance is attached and the policy source is the checkpoint
    assert pred["engine"]["checkpointFile"] == "best.pt"
    assert pred["thresholdPolicy"]["source"] == "checkpoint (validation-selected)"
    assert pred["engine"]["engineMode"] == "real-model"


def test_backbone_mode_has_no_provenance(tmp_path):
    engine = _build_engine(str(tmp_path))  # empty dir → no checkpoint
    assert engine is not None
    assert engine["num_classes"] == 1000
    meta = _engine_meta(engine)
    assert meta["engineMode"] == "backbone-live"
    assert meta["source"] == "pytorch-backbone"
    assert meta["weightsLoaded"] is False
    # provenance fields only exist for real checkpoints — never fabricated
    assert "checkpointFile" not in meta
    assert "trainedClasses" not in meta


def test_last_pt_fallback_when_no_best(tmp_path):
    _fake_checkpoint(str(tmp_path / "last.pt"))
    engine = _build_engine(str(tmp_path))
    assert os.path.basename(engine["checkpoint"]) == "last.pt"
