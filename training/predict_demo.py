"""
MedVision AI — run real predictions on held-out images with a trained checkpoint.

Usage:
  python training/predict_demo.py --checkpoint training/checkpoints/best.pt \\
      --config training/configs/train.yaml
  python training/predict_demo.py --checkpoint ... --images a.png b.png

Pipeline per image:
  image → preprocessing (Resize 256 / CenterCrop 224 / ImageNet normalize)
       → DenseNet-121 → sigmoid probabilities → frozen thresholds → labels
       → Grad-CAM (top predicted class) → overlay saved to training/results/demo/

This is a verification/demo tool. It never modifies the model. If no checkpoint
(or no dataset) is available, it reports exactly that and exits without output.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from dataset import DEFAULT_LABELS, get_transform, load_split_csv  # noqa: E402

INFERENCE_TRANSFORM = get_transform("val")  # identical to backend/app/main.py


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="MedVision AI — real prediction + Grad-CAM demo")
    parser.add_argument("--checkpoint", required=True, help="path to a trained .pt checkpoint")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "train.yaml"))
    parser.add_argument("--images", nargs="*", default=None, help="explicit image paths (overrides test split)")
    parser.add_argument("--out", default="training/results/demo")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--save-overlays", action="store_true", default=True, help="save Grad-CAM overlays")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"\n[data] Prediction demo not executed — checkpoint not found: {args.checkpoint}\n")
        print("       Train first:  python training/train.py --config training/configs/train.yaml\n")
        sys.exit(0)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("metadata") or {}
    label_names = list(meta.get("label_names") or DEFAULT_LABELS)
    num_classes = ckpt["state_dict"]["classifier.weight"].shape[0]
    if num_classes < len(label_names):
        label_names = label_names[:num_classes]
    thresholds = (meta.get("thresholds") or {lab: 0.5 for lab in label_names})
    unavailable = set(meta.get("unavailable_classes") or [])
    print(f"[model] {args.checkpoint}")
    print(f"        classes: {num_classes} · trained: {[l for l in label_names if l not in unavailable]} "
          f"· unavailable: {sorted(unavailable)}")
    print(f"        thresholds: {json.dumps(thresholds)}")
    print(f"        checkpoint epoch={ckpt.get('epoch')} kind={ckpt.get('kind')} "
          f"val_macro_auc={ckpt.get('val_macro_auc')}")

    device = "cpu"
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] Using {'CUDA' if device == 'cuda' else 'MPS' if device == 'mps' else 'CPU'}")

    model = torchvision.models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()})
    model.to(device)
    model.eval()

    # Input images: explicit paths, else the held-out test split.
    if args.images:
        paths = args.images
    else:
        cfg = {}
        if os.path.exists(args.config):
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        ds = cfg.get("dataset", {})
        test_records = load_split_csv(os.path.join(str(ds.get("splits_dir") or "training/splits"), "test.csv"),
                                      str(ds.get("images_dir")))
        if not test_records:
            print("[data] No explicit --images and no test split found — nothing to predict.")
            sys.exit(0)
        paths = [r["image_path"] for r in test_records][:10]
    print(f"[data] images: {len(paths)}")

    os.makedirs(args.out, exist_ok=True)
    results = []
    for idx, path in enumerate(paths):
        if not os.path.exists(path):
            print(f"  [skip] {path} (missing)")
            continue
        img = Image.open(path).convert("RGB")
        x = INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        preds = {lab: (float(probs[c]) >= float(thresholds[lab])) for c, lab in enumerate(label_names)}
        top_c = int(np.argmax(probs))
        top_lab, top_prob = label_names[top_c], float(probs[top_c])
        results.append({"image": path, "probabilities": {lab: round(float(probs[c]), 4) for c, lab in enumerate(label_names)},
                        "predicted_at_threshold": {lab: bool(v) for lab, v in preds.items() if v and lab not in unavailable},
                        "top": top_lab, "top_probability": round(top_prob, 4)})
        print(f"\n  [{idx}] {os.path.basename(path)}")
        for c, lab in enumerate(label_names):
            flag = "◉" if preds[lab] and lab not in unavailable else ("◦" if lab not in unavailable else "—")
            mark = " (unavailable class)" if lab in unavailable else ""
            print(f"       {lab:<18} p={probs[c]:.4f}  thr={thresholds[lab]:.2f}  {flag}{mark}")
        print(f"       TOP: {top_lab} (p={top_prob:.4f})")

        # Grad-CAM for the top predicted class (real gradients through the head).
        if args.save_overlays and top_lab not in unavailable and top_c < num_classes:
            try:
                from app.models.gradcam import gradcam_map, make_overlay
                from torchvision.transforms import functional as F
                target_layer = model.features.denseblock4.denselayer16.conv2
                cam = gradcam_map(model, x, top_c, target_layer)
                overlay = make_overlay(img, cam, alpha=0.55, name="jet")
                out_path = os.path.join(args.out, f"demo_{idx}_{top_lab.replace(' ', '_')}_gradcam.png")
                overlay.save(out_path)
                print(f"       Grad-CAM overlay saved → {out_path}")
            except Exception as exc:
                print(f"       Grad-CAM failed: {exc}")

    with open(os.path.join(args.out, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "device": device, "results": results}, f, indent=2)
    print(f"\n[out] predictions.json + overlays → {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
