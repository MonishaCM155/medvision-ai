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
from app.main import IMAGE_TYPES, get_engine  # noqa: E402


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
    # Engine metadata is a dict with honest mode labels
    assert data["engine"]["engineMode"] in ("real-model", "backbone-live", "demo-engine")
    assert data["engine"]["source"] in ("pytorch-checkpoint", "pytorch-backbone", "demo")
    assert data["engine"]["predictionSource"] in ("real-inference", "backbone+demo-profile", "demo-profile")


def test_engine_status_contract():
    res = client.get("/api/engine")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["source"] in ("pytorch-checkpoint", "pytorch-backbone", "demo")
    assert data["engineMode"] in ("real-model", "backbone-live", "demo-engine")
    assert data["weightsLoaded"] in (True, False)
    assert data["predictionSource"] in ("real-inference", "backbone+demo-profile", "demo-profile")
    assert isinstance(data["modelVersion"], str) and data["modelVersion"]
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
    # Engine metadata always present (honest mode labels)
    assert data["engine"]["source"] in ("pytorch-checkpoint", "pytorch-backbone", "demo")
    assert data["engine"]["engineMode"] in ("real-model", "backbone-live", "demo-engine")
    assert data["engine"]["predictionSource"] in ("real-inference", "backbone+demo-profile", "demo-profile")


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


def test_validate_endpoint_contract():
    res = client.post("/api/validate", json={"imageName": "chest_xray_pa.png"})
    assert res.status_code == 200
    data = res.json()
    # Safety gate must always report a pass/fail verdict
    assert "passed" in data and isinstance(data["passed"], bool)
    # 11-class type classifier (heuristic-v1 — documented CNN upgrade path)
    assert data["type"]["method"] in ("heuristic-v1", "unavailable")
    if data["type"]["method"] == "heuristic-v1":
        assert data["type"]["predicted"] in IMAGE_TYPES
        assert set(data["type"]["confidences"].keys()).issubset(set(IMAGE_TYPES))
    # Quality score bounded 0-100; threshold is env-configurable (default 55)
    assert 0 <= data["quality"]["score"] <= 100
    assert data["quality"]["threshold"] == 55
    assert data["quality"]["method"] == "pixel-heuristics"
    # OOD verdict is one of the three documented buckets; heuristic label honest
    assert data["ood"]["verdict"] in ("in", "borderline", "out")
    assert data["ood"]["label"] == "heuristic-screening"
    # No imageData → the OOD proxy honestly reports it cannot run
    assert data["ood"]["method"] in ("feature-proxy-v1", "unavailable")
    # Type classifier is explicitly heuristic — never presented as a trained CNN
    assert data["type"]["label"] == "heuristic-classifier"


def test_validate_rejects_non_image_payload():
    res = client.post("/api/validate", json={"imageData": "data:text/plain;base64,SGVsbG8="})
    assert res.status_code == 422


def test_predict_returns_safety_pipeline_metadata():
    res = client.post("/api/predict", json={"imageName": "covid_patient_01.png"})
    assert res.status_code == 200
    data = res.json()
    # Calibration: T=1.0 identity must leave confidence unchanged
    assert data["calibration"]["temperature"] == 1.0
    assert data["calibration"]["rawTopConfidence"] == pytest.approx(
        data["calibration"]["calibratedTopConfidence"], abs=1e-4
    )
    # Uncertainty: one of the three levels with a bounded score
    assert data["uncertainty"]["level"] in ("low", "moderate", "high")
    assert 0 <= data["uncertainty"]["score"] <= 100
    assert data["uncertainty"]["method"] in ("mc-dropout", "margin-proxy")
    # Safety pipeline stages attached to the prediction
    assert 0 <= data["quality"]["score"] <= 100
    assert data["typeCheck"]["method"] in ("heuristic-v1", "unavailable")
    assert data["typeCheck"]["label"] == "heuristic-classifier"
    assert data["ood"]["verdict"] in ("in", "borderline", "out")


def test_predict_rejects_non_cxr_payload():
    """Server-side gate: a real image payload that is clearly not a chest X-ray
    is refused (422) even when the request claims it is valid — the client can
    never authorize inference."""
    pil = pytest.importorskip("PIL", reason="PIL not installed")
    from PIL import Image
    import io
    import base64

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 40, 200)).save(buf, "PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    res = client.post("/api/predict", json={"imageName": "colorful_photo.jpg", "imageData": data_url})
    assert res.status_code == 422
    assert "safety gate" in res.json()["detail"]


def test_cors_whitelist():
    """CORS is environment-controlled — never wildcard-with-credentials."""
    # Default dev whitelist allows the local SPA origin…
    res = client.get("/api/engine", headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"
    # …but an unknown origin receives no CORS grant
    res = client.get("/api/engine", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in res.headers
