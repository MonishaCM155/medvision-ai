#!/usr/bin/env python
"""
MedVision AI — reproducible end-to-end smoke test for the FastAPI ML engine.

Usage:
    python tests/e2e_smoke.py
    MEDVISION_SKIP_PRETRAINED=1 python tests/e2e_smoke.py   # random-init backbone (CI-fast)

Verifies the full chain on CPU (CUDA/MPS used automatically when available):

    1. Engine starts (lazy PyTorch load, honest mode reported)
    2. Engine health works
    3. A synthetic chest-X-ray-like image is generated
    4. Image validation works (structured per-check PASS/WARN/FAIL report)
    5. Prediction works (real forward pass, 10-class multi-label output)
    6. Explainability works (real activation map + overlay + peak region)
    7. Report generation works (findings/impression/recommendations/disclaimer)
    8. Similar-case retrieval works (real feature embeddings + cosine similarity)

Every assertion is contract-based — it verifies shape and honesty labels, never
fabricated accuracy values.
"""

import base64
import io
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

PASSED, FAILED = 0, 0


def ok(cond, label, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  \u2713 {label}")
    else:
        FAILED += 1
        print(f"  \u2717 {label} {('- ' + str(extra)) if extra else ''}")


def make_synthetic_cxr(size=512, seed=0):
    """A synthetic image with the thoracic signature the safety gate expects:
    bright mediastinal band down the middle, dark flanks, bright upper zone."""
    img = Image.new("L", (size, size), 90)
    d = ImageDraw.Draw(img)
    d.rectangle([int(size * 0.40), int(size * 0.08), int(size * 0.60), int(size * 0.92)], fill=175)
    d.rectangle([int(size * 0.25), int(size * 0.05), int(size * 0.75), int(size * 0.32)], fill=135)
    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.default_rng(seed).normal(0, 6, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("\nMedVision AI — FastAPI engine end-to-end smoke test\n")

    # 1. Engine starts
    from app.main import app, get_engine, DISEASE_LABELS  # noqa: E402
    from fastapi.testclient import TestClient

    client = TestClient(app)
    engine = get_engine()
    mode = "backbone-live (pretrained)" if engine and engine.get("num_classes", 0) == 1000 else \
        "real-model (fine-tuned)" if engine and engine.get("num_classes", 0) in (10, 14) else "demo-engine"
    ok(engine is not None, f"engine started — {mode}", "engine is None (torch missing?)")
    device = str((engine or {}).get("device") or "cpu")
    print(f"  using device: {device}")

    # 2. Engine health
    r = client.get("/api/health")
    ok(r.status_code == 200 and r.json()["status"] == "ok", "GET /api/health -> 200 ok")
    r = client.get("/api/engine")
    body = r.json()
    ok(r.status_code == 200 and body["status"] == "ready" and body["source"] in
       ("pytorch-checkpoint", "pytorch-backbone", "demo"), "GET /api/engine reports honest mode")

    # 3. Sample X-ray loads + 4. validation
    cxr = make_synthetic_cxr()
    data_url = to_data_url(cxr)
    r = client.post("/api/validate", json={"imageName": "smoke_cxr.png", "imageData": data_url})
    v = r.json()
    ok(r.status_code == 200 and v.get("passed") is True, "validation passes for synthetic CXR",
       v.get("message"))
    checks = v.get("checks", [])
    ok(len(checks) == 8, f"structured validation has 8 checks (got {len(checks)})")
    ok(all(c.get("status") in ("pass", "warn", "fail") and c.get("detail") for c in checks),
       "every check carries a PASS/WARN/FAIL status + reason")
    keys = {c["key"] for c in checks}
    ok({"format", "resolution", "grayscale", "contrast", "brightness", "sharpness", "orientation", "chest_xray"} <= keys,
       "all 8 documented check keys present", sorted(keys))
    ok(v.get("valid") is True and isinstance(v.get("quality_score"), (int, float)),
       "valid/quality_score contract present")

    # 5. Prediction
    r = client.post("/api/predict", json={"imageName": "smoke_cxr.png", "imageData": data_url})
    p = r.json()
    ok(r.status_code == 200, "POST /api/predict -> 200", p.get("detail"))
    ok(len(p.get("diseases", [])) == len(DISEASE_LABELS), "10-class multi-label output")
    ok(0 <= p["topConfidence"] <= 1 and p["topDiagnosis"], "top prediction + confidence present")
    ok(all("threshold" in d for d in p["diseases"]), "per-class decision threshold present")
    ok(p["report"]["findings"] and p["report"]["impression"] and p["report"]["disclaimer"],
       "structured report generated (findings/impression/disclaimer)")

    # 6. Explainability (real maps when the engine is up)
    expl = p.get("explainability") or {}
    ok(expl.get("computed") is True, "explainability computed", expl.get("method"))
    ok(bool(p.get("heatmapUrl")) and bool(p.get("heatmapOverlayUrl")), "real heatmap + overlay data URLs present")
    ok(bool(p.get("gradCamRegions")) and p["gradCamRegions"][0].get("bbox"),
       "peak-activation region with real bbox", p["gradCamRegions"][0] if p.get("gradCamRegions") else None)
    ok(len(p.get("validationChecks", [])) == 8, "prediction carries server validation checks")

    # 7. Similar-case retrieval (real embeddings)
    refs = [
        {"id": "ref-a", "title": "Reference A", "label": "Pneumonia", "imageData": to_data_url(make_synthetic_cxr(seed=1))},
        {"id": "ref-b", "title": "Reference B", "label": "Cardiomegaly", "imageData": to_data_url(make_synthetic_cxr(seed=2))},
    ]
    r = client.post("/api/similar-cases", json={"queryImage": data_url, "references": refs})
    sim = r.json()
    ok(r.status_code == 200 and len(sim.get("cases", [])) == 2, "similar-cases returns ranked cohort", sim.get("detail"))
    ok(all(0 <= c["similarity"] <= 1 for c in sim.get("cases", [])), "similarities are real cosine values in [0,1]")
    ok("not diagnostic evidence" in (sim.get("disclaimer") or "").lower(), "similarity disclaimer present")

    print(f"\n{'-' * 56}\nRESULT: {PASSED} passed · {FAILED} failed\n")
    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    main()
