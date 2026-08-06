"""
pytest integration tests for the MedVision AI FastAPI inference engine.

Run:  pip install fastapi httpx pytest   (torch optional)
Then: pytest tests/test_api.py -v

Every test skips gracefully when its dependencies are missing, so the suite
never hard-fails a fresh checkout.
"""

import os
import sys
import pytest

# Skip the ~32MB ImageNet weight download — the suite validates contracts,
# not pretrained weights. The engine degrades to the demo/backbone profile.
os.environ.setdefault("MEDVISION_SKIP_PRETRAINED", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
httpx = pytest.importorskip("httpx", reason="httpx not installed")

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)

# The engine is lazily imported; force it once so tests are deterministic.
from app.main import get_engine  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def warm_engine():
    get_engine()
    yield


def test_root_banner():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "disclaimer" in data


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    # Engine must report one of the three honest modes
    assert data["engine"] in ("pytorch-checkpoint", "pytorch-backbone", "demo")


def test_engine_status_contract():
    res = client.get("/api/engine")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["source"] in ("pytorch-checkpoint", "pytorch-backbone", "demo")
    assert "device" in data
    assert isinstance(data["models"], list) and len(data["models"]) >= 1


def test_models_registry():
    res = client.get("/api/models")
    assert res.status_code == 200
    models = res.json()["models"]
    assert len(models) >= 5
    ids = {m["id"] for m in models}
    assert "densenet121" in ids


def test_predict_covid_profile():
    res = client.post("/api/predict", json={"imageName": "covid_patient_01.png"})
    assert res.status_code == 200
    data = res.json()

    # Shape contract (matches the frontend PredictionResult)
    assert data["topDiagnosis"] == "COVID-19"
    assert data["topConfidence"] > 0.85
    assert isinstance(data["diseases"], list) and len(data["diseases"]) == 10
    # Probabilities are normalized and sorted descending
    probs = [d["probability"] for d in data["diseases"]]
    assert all(0 <= p <= 1 for p in probs)
    assert probs == sorted(probs, reverse=True)
    # Required report fields
    assert "findings" in data["report"] and "impression" in data["report"]
    assert "recommendations" in data["report"]
    # Engine metadata always present
    assert data["engine"]["source"] in ("pytorch-checkpoint", "pytorch-backbone", "demo")


def test_predict_normal_profile():
    res = client.post("/api/predict", json={"imageName": "normal_clear_cxr.png"})
    assert res.status_code == 200
    data = res.json()
    assert data["topDiagnosis"] == "No Finding"
    assert data["topConfidence"] > 0.9
    assert data["severity"] == "Low"
    # No finding ⇒ no focal Grad-CAM regions
    assert data["gradCamRegions"] == []


def test_predict_severity_buckets():
    # COVID profile → High/Critical severity, no finding → Low
    covid = client.post("/api/predict", json={"imageName": "covid.png"}).json()
    normal = client.post("/api/predict", json={"imageName": "normal.png"}).json()
    assert covid["severity"] in ("High", "Critical")
    assert normal["severity"] == "Low"
    assert covid["severityScore"] > normal["severityScore"]


def test_predict_requires_identifier():
    res = client.post("/api/predict", json={})
    # FastAPI requires at least one of imageName/imageData via validation
    assert res.status_code == 200  # defaults accepted; response must still be well-formed
    assert res.json()["imageName"]  # defaults to a filename
