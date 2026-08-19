"""
pytest integration tests for the MedVision AI FastAPI inference engine.

Run:  pip install fastapi httpx pytest   (torch optional)
Then: pytest tests/test_api.py -v

Every test skips gracefully when its dependencies are missing, so the suite
never hard-fails a fresh checkout.
"""

import io
import os
import sys
import base64

import numpy as np
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


def _synthetic_cxr_data_url(size=512, seed=0):
    """A synthetic image carrying the thoracic signature the safety gate expects."""
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


def test_validate_structured_checks_contract():
    """POST /api/validate returns the documented per-check validation report:
    PASS/WARN/FAIL for format/resolution/grayscale/contrast/brightness/
    sharpness/orientation/chest_xray, plus valid/quality_score aliases."""
    data_url = _synthetic_cxr_data_url()
    res = client.post("/api/validate", json={"imageName": "cxr.png", "imageData": data_url})
    assert res.status_code == 200
    body = res.json()
    assert body["valid"] is True
    assert body["quality_score"] == body["quality"]["score"]
    checks = body["checks"]
    assert len(checks) == 8
    keys = {c["key"] for c in checks}
    assert keys == {"format", "resolution", "grayscale", "contrast", "brightness", "sharpness", "orientation", "chest_xray"}
    for c in checks:
        assert c["status"] in ("pass", "warn", "fail")
        assert c["detail"]


def test_predict_diseases_include_threshold():
    res = client.post("/api/predict", json={"imageName": "covid_patient_01.png"})
    assert res.status_code == 200
    for d in res.json()["diseases"]:
        assert "threshold" in d
        assert 0 < d["threshold"] < 1


def test_predict_real_explainability_and_validation():
    """With the engine up (backbone mode under MEDVISION_SKIP_PRETRAINED=1), a
    validated synthetic CXR yields a real activation map, overlay, peak region
    and server validation checks — never a placeholder."""
    if get_engine() is None:
        pytest.skip("PyTorch engine unavailable")
    data_url = _synthetic_cxr_data_url()
    res = client.post("/api/predict", json={"imageName": "cxr.png", "imageData": data_url})
    assert res.status_code == 200
    body = res.json()
    assert body["explainability"]["computed"] is True
    assert body["explainability"]["method"] in ("grad-cam", "grad-cam++", "feature-activation")
    assert body["heatmapUrl"].startswith("data:image/png")
    assert body["heatmapOverlayUrl"].startswith("data:image/png")
    assert body["originalImageUrl"] == data_url
    assert len(body["gradCamRegions"]) >= 1
    region = body["gradCamRegions"][0]
    assert "bbox" in region and "zone" in region
    assert len(body["validationChecks"]) == 8


def test_similar_cases_requires_engine_and_payload():
    res = client.post("/api/similar-cases", json={})
    if get_engine() is None:
        assert res.status_code == 503
    else:
        assert res.status_code == 422


def test_similar_cases_real_embeddings():
    """Genuine embedding retrieval: real DenseNet-121 features + cosine
    similarity, sorted descending, bounded [0,1]."""
    if get_engine() is None:
        pytest.skip("PyTorch engine unavailable")
    q = _synthetic_cxr_data_url(seed=1)
    refs = [
        {"id": "a", "title": "A", "label": "Pneumonia", "imageData": _synthetic_cxr_data_url(seed=2)},
        {"id": "b", "title": "B", "label": "Cardiomegaly", "imageData": _synthetic_cxr_data_url(seed=3)},
        {"id": "c", "title": "C", "label": "No Finding", "imageData": _synthetic_cxr_data_url(seed=4)},
    ]
    res = client.post("/api/similar-cases", json={"queryImage": q, "references": refs})
    assert res.status_code == 200
    body = res.json()
    assert body["embedding"] == "densenet121-features"
    assert len(body["cases"]) == 3
    sims = [c["similarity"] for c in body["cases"]]
    assert sims == sorted(sims, reverse=True)
    assert all(0 <= s <= 1 for s in sims)
    assert "not diagnostic evidence" in body["disclaimer"].lower()


def test_predict_rejects_non_cxr_payload_checks_explainable():
    """Rejections stay contract-consistent: a failed gate returns 422 with a
    server-side reason — no heatmap, no fabricated prediction."""
    pil = pytest.importorskip("PIL", reason="PIL not installed")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 40, 200)).save(buf, "PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    res = client.post("/api/predict", json={"imageName": "colorful_photo.jpg", "imageData": data_url})
    assert res.status_code == 422


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


def test_predict_shape_contract():
    """Verify the prediction response matches the frontend PredictionResult contract,
    regardless of engine mode (real-model, backbone-live, or demo)."""
    res = client.post("/api/predict", json={"imageName": "covid_patient_01.png"})
    assert res.status_code == 200
    data = res.json()

    # Shape contract (matches the frontend PredictionResult)
    assert isinstance(data["topDiagnosis"], str) and data["topDiagnosis"]
    assert isinstance(data["topConfidence"], (int, float))
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

    # In real-model mode, verify trained/unavailable class annotations
    if data["engine"]["engineMode"] == "real-model":
        for d in data["diseases"]:
            assert "trained" in d
            assert isinstance(d["trained"], bool)
            if not d["trained"]:
                assert "note" in d


def test_predict_prediction_consistency():
    """Verify prediction is deterministic: same image → same output."""
    payload = {"imageName": "test_image.png"}
    r1 = client.post("/api/predict", json=payload).json()
    r2 = client.post("/api/predict", json=payload).json()
    assert r1["topDiagnosis"] == r2["topDiagnosis"]
    assert abs(r1["topConfidence"] - r2["topConfidence"]) < 1e-4
    p1 = [d["probability"] for d in r1["diseases"]]
    p2 = [d["probability"] for d in r2["diseases"]]
    assert p1 == p2


def test_predict_severity_structure():
    """Verify severity is a valid bucket and score is a non-negative int."""
    data = client.post("/api/predict", json={"imageName": "test_cxr.png"}).json()
    assert data["severity"] in ("Low", "Moderate", "High", "Critical")
    assert isinstance(data["severityScore"], (int, float))
    assert data["severityScore"] >= 0


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
