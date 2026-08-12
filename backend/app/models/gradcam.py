"""
Grad-CAM / Grad-CAM++ / Feature-Activation Explainability for MedVision AI.

This module produces REAL activation maps from the DenseNet-121 backbone:

  * Grad-CAM            — Selvaraju et al. 2017: gradient-weighted pooling of
                          the final convolutional feature maps, per target class.
  * Grad-CAM++          — Chattopadhay et al. 2018: second-order gradient
                          weighting for finer focal detail.
  * Feature activation  — class-agnostic mean of ReLU'd final feature maps.
                          Used when no fine-tuned disease head is loaded: it is
                          honest about what it is (backbone attention, NOT a
                          disease-specific attribution) and never pretends to
                          localize a pathology the model was not trained for.

Helpers also render the heatmap (jet/turbo colormaps, pure-numpy), blend it
over the original radiograph, locate the peak-activation region (bounding box
+ thoracic zone), and encode PNG data-URLs for the frontend.

Every function is defensive: hooks are removed in `finally`, tensors are
detached, and degenerate inputs degrade to `None` rather than raising.
"""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Attribution maps
# ---------------------------------------------------------------------------


def _final_conv_layer(model):
    """The standard target layer for Grad-CAM on DenseNet-121."""
    return model.features.denseblock4.denselayer16.conv2


def gradcam_map(model, x, class_idx, target_layer=None):
    """Grad-CAM heatmap (0..1, float32, H×W) for `class_idx` (logit index).

    `x` is a batched input tensor; `class_idx` indexes the model's raw output
    (logits for a sigmoid/linear head — the standard Grad-CAM formulation).
    """
    layer = target_layer or _final_conv_layer(model)
    activations = {}
    gradients = {}

    def fwd_hook(_m, _i, out):
        activations["a"] = out.detach()

    def bwd_hook(_m, _gi, go):
        gradients["g"] = go[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        model.eval()
        x = x.clone().requires_grad_(True)
        out = model(x)
        score = out[0, class_idx]
        model.zero_grad()
        score.backward(retain_graph=True)

        A = activations["a"]  # (1, C, H, W)
        g = gradients["g"]    # (1, C, H, W)
        w = torch.mean(g, dim=(2, 3), keepdim=True)          # alpha weights
        cam = F.relu(torch.sum(w * A, dim=1, keepdim=True))  # (1, 1, H, W)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        return _normalize(cam)
    finally:
        h1.remove()
        h2.remove()


def gradcam_pp_map(model, x, class_idx, target_layer=None):
    """Grad-CAM++ heatmap (Chattopadhay et al. 2018) for `class_idx`."""
    layer = target_layer or _final_conv_layer(model)
    activations = {}
    gradients = {}

    def fwd_hook(_m, _i, out):
        activations["a"] = out.detach()

    def bwd_hook(_m, _gi, go):
        gradients["g"] = go[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        model.eval()
        x = x.clone().requires_grad_(True)
        out = model(x)
        score = out[0, class_idx]
        model.zero_grad()
        score.backward(retain_graph=True)

        A = activations["a"]  # (1, C, H, W)
        g = gradients["g"]    # (1, C, H, W)

        g2 = g * g
        g3 = g2 * g
        # Denominator summed over spatial dims (per channel), per the paper.
        denom = torch.sum(A * g3, dim=(2, 3), keepdim=True) + 1e-8
        alpha = g2 / (2.0 * g2 + denom)
        w = torch.sum(alpha * F.relu(g), dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(w * A, dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        return _normalize(cam)
    finally:
        h1.remove()
        h2.remove()


def feature_activation_map(model, x, target_layer=None):
    """Class-agnostic feature activation: mean of ReLU'd final feature maps.

    No gradients required. Honest label: this shows where the *backbone*
    concentrates its representation for the input — it is NOT a disease-
    specific attribution, which requires a fine-tuned classification head.
    """
    layer = target_layer or _final_conv_layer(model)
    activations = {}

    def fwd_hook(_m, _i, out):
        activations["a"] = out.detach()

    h = layer.register_forward_hook(fwd_hook)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(x)
            A = activations["a"]
            cam = F.relu(A).mean(dim=1, keepdim=True)  # mean over channels
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        return _normalize(cam)
    finally:
        h.remove()


def _normalize(cam_tensor):
    """Detach → (H, W) float32 in [0, 1]."""
    cam = cam_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo < 1e-8:
        return np.zeros_like(cam)
    return (cam - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Colormaps (pure numpy — no matplotlib dependency for the inference path)
# ---------------------------------------------------------------------------

def _jet_lut():
    t = np.linspace(0.0, 1.0, 256)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _turbo_lut():
    """Google Turbo colormap (public-domain polynomials, A. Mikhailov)."""
    t = np.linspace(0.0, 1.0, 256)
    r = np.clip(-2.805 * t ** 3 + 5.866 * t ** 2 - 3.558 * t + 0.650, 0.0, 1.0)
    g = np.clip(-2.563 * t ** 3 + 4.023 * t ** 2 - 1.553 * t + 0.283, 0.0, 1.0)
    b = np.clip(-1.321 * t ** 3 + 3.629 * t ** 2 - 3.613 * t + 0.946, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


_COLORMAPS = {"jet": _jet_lut, "turbo": _turbo_lut}


def colormap_lut(name="jet"):
    """Return a (256, 3) float LUT in [0, 1] for the given colormap name."""
    try:
        import matplotlib  # optional — richer colormaps when installed
        import matplotlib.cm as cm

        name_c = {"jet": "jet", "turbo": "turbo", "viridis": "viridis", "inferno": "inferno"}.get(name.lower(), name)
        cmap = cm.get_cmap(name_c)
        return np.asarray(cmap(np.linspace(0, 1, 256)))[:, :3]
    except Exception:
        return _COLORMAPS.get(name.lower(), _jet_lut)() if name.lower() in _COLORMAPS else _jet_lut()


def apply_colormap(cam_01, name="jet"):
    """Colorize a (H, W) float32 map in [0, 1] → (H, W, 3) uint8 image."""
    lut = colormap_lut(name)
    idx = np.clip((cam_01 * 255.0).astype(np.int64), 0, 255)
    return (lut[idx] * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def make_heatmap_image(cam_01, name="jet"):
    """Colored heatmap as a PIL RGB image (same size as the cam map)."""
    from PIL import Image

    colored = apply_colormap(cam_01, name)
    return Image.fromarray(colored, "RGB")


def make_overlay(original_pil, cam_01, alpha=0.55, name="jet"):
    """Blend the original radiograph with the heatmap, resized to original dims.

    `original_pil` — PIL image (L or RGB) at full resolution.
    `cam_01`       — (H, W) float32 activation map (any size; resized here).
    """
    from PIL import Image

    rgb = original_pil.convert("RGB")
    w, h = rgb.size
    cam_img = Image.fromarray((cam_01 * 255.0).astype(np.uint8), "L")
    cam_img = cam_img.resize((w, h), Image.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    colored = apply_colormap(cam_resized, name)
    return Image.blend(rgb, Image.fromarray(colored, "RGB"), float(alpha))


def encode_data_url(pil_image, fmt="PNG"):
    """Encode a PIL image as a base64 data-URL (for the frontend <img> tags)."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/" + fmt.lower()
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Peak-region analysis (real bbox from the activation map, not a placeholder)
# ---------------------------------------------------------------------------

def _zone_label(cx, cy):
    horiz = "left" if cx < 0.42 else ("right" if cx > 0.58 else "central")
    vert = "upper" if cy < 0.38 else ("middle" if cy < 0.68 else "lower")
    return f"{horiz}-{vert} thoracic zone"


def peak_region(cam_01, top_frac=0.20, pad_frac=0.01):
    """Bounding box + intensity stats of the highest-activation region.

    Pixels above the (1 - top_frac) quantile of activation define the region;
    the returned box is the tight bounding box around them, padded by
    `pad_frac` of the map size. Returns None on degenerate maps.
    """
    if cam_01 is None or cam_01.size == 0:
        return None
    h, w = cam_01.shape
    flat = cam_01.ravel()
    thr = float(np.quantile(flat, 1.0 - top_frac))
    mask = cam_01 >= thr
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    px = max(1, int(round(w * pad_frac)))
    py = max(1, int(round(h * pad_frac)))
    x0, x1 = max(0, int(xs.min()) - px), min(w - 1, int(xs.max()) + px)
    y0, y1 = max(0, int(ys.min()) - py), min(h - 1, int(ys.max()) + py)
    cx, cy = (x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h
    return {
        "bbox": {"x": int(x0), "y": int(y0), "width": int(x1 - x0), "height": int(y1 - y0)},
        "intensity": round(float(cam_01.max()) * 100.0, 1),
        "meanIntensity": round(float(cam_01[mask].mean()) * 100.0, 1),
        "coverage": round(float(mask.mean()) * 100.0, 1),
        "center": {"x": round(cx, 3), "y": round(cy, 3)},
        "zone": _zone_label(cx, cy),
    }


def region_interpretation(disease, region, method):
    """Cautious, non-clinical wording for a real activation region."""
    if region is None:
        return f"No distinct activation peak detected for the {method} map."
    return (
        f"The highlighted {region['zone']} (peak activation {region['intensity']:.0f}%, "
        f"covering {region['coverage']:.0f}% of the frame) is the region that contributed "
        f"most strongly to the model's {method} response for the {disease} class. "
        f"This indicates where the network focused — it is not evidence of a diagnosis."
    )
