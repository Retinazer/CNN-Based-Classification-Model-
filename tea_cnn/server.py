"""
FastAPI inference server for the trained tea-leaf disease classifier (TDD).

Loads checkpoints/best_model.pt (MobileNetV2, 9 classes) and exposes:
  POST /predict  - multipart image upload ->
                   JSON {label, confidence, top3, severity, summary, action,
                         heatmap (base64 PNG attention overlay), focus_region}
  GET  /health   - liveness check
  GET  /         - serves the mobile web app (mobile_app/index.html)

The "heatmap"/"focus_region" fields come from a real Grad-CAM pass over the
model's last conv block -- they show where the network actually weighted
its decision, not a canned explanation. This is approximate evidence of
*where* the model looked, not a verified clinical cause.

Run:  python3 server.py
Then open http://<this-machine's-LAN-IP>:8000 on your phone (same Wi-Fi).
"""
import base64
import io
import os

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from common import BEST_CHECKPOINT_PATH, IMG_SIZE, PROJECT_DIR
from train import build_model, eval_transform

APP_DIR = os.path.join(PROJECT_DIR, "mobile_app")

ADVICE = {
    "healthy": {
        "severity": "none",
        "summary": "No disease signs detected.",
        "action": "No treatment needed — continue routine weekly scouting during the flush season.",
    },
    "algal_spot": {
        "severity": "low",
        "summary": "Algal leaf spot — rust-colored, slightly raised spots from parasitic algae, favored by humid, shaded conditions.",
        "action": "Improve canopy airflow by pruning overcrowded shoots, and apply a copper-based fungicide if spread continues.",
    },
    "anthracnose": {
        "severity": "medium",
        "summary": "Anthracnose — irregular brown-black lesions with concentric rings, spreads fast in wet weather.",
        "action": "Remove and destroy infected leaves, avoid overhead irrigation, and apply a protectant fungicide within 48 hours.",
    },
    "bird_eye_spot": {
        "severity": "medium",
        "summary": "Bird's eye spot — small circular lesions with a pale center and dark ring, resembling a bird's eye.",
        "action": "Prune affected shoots and apply a recommended fungicide; monitor for spread to adjacent bushes.",
    },
    "brown_blight": {
        "severity": "medium",
        "summary": "Brown blight — large brown necrotic patches, often starting at the leaf tip or margin.",
        "action": "Remove heavily affected leaves, improve drainage around the bush, and apply a copper-based fungicide.",
    },
    "gray_blight": {
        "severity": "medium",
        "summary": "Gray blight — gray-centered lesions with a dark brown margin, common in warm, humid conditions.",
        "action": "Prune for airflow, remove fallen infected leaves from the ground, and apply a fungicide rotation.",
    },
    "helopeltis": {
        "severity": "high",
        "summary": "Tea mosquito bug (Helopeltis) damage — dark necrotic feeding spots and curled young leaves, can halt flush growth within days.",
        "action": "Spray a recommended insecticide rotation on the affected block today and inspect neighboring rows — this pest spreads fast in warm, humid weather.",
    },
    "red_spot": {
        "severity": "low",
        "summary": "Red leaf spot — reddish-brown circular lesions, usually a minor fungal issue.",
        "action": "Monitor weekly; apply a fungicide only if lesion coverage increases beyond a few spots per leaf.",
    },
    "white_spot": {
        "severity": "low",
        "summary": "White spot — small white-centered lesions with a dark border.",
        "action": "Usually minor; improve airflow and monitor. Treat with fungicide only if spreading rapidly.",
    },
}

app = FastAPI(title="TDD - Tea Disease Detector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
classes = checkpoint["classes"]
model = build_model(len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded model on {device}. Classes: {classes}")

# --- Grad-CAM hooks on the last conv block of MobileNetV2's feature extractor ---
_gradcam_target = model.features[-1]
_activations = {}
_gradients = {}


def _forward_hook(module, inp, out):
    _activations["value"] = out


def _backward_hook(module, grad_in, grad_out):
    _gradients["value"] = grad_out[0]


_gradcam_target.register_forward_hook(_forward_hook)
_gradcam_target.register_full_backward_hook(_backward_hook)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# simple blue -> green -> yellow -> red colormap, no extra dependency needed
_CMAP_STOPS = np.array(
    [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ],
    dtype=np.float32,
)


def _colorize(gray01: np.ndarray) -> np.ndarray:
    """gray01: HxW float array in [0,1] -> HxWx3 uint8 using a fixed colormap."""
    n = len(_CMAP_STOPS) - 1
    scaled = np.clip(gray01, 0, 1) * n
    idx = np.clip(scaled.astype(int), 0, n - 1)
    frac = (scaled - idx)[..., None]
    c0 = _CMAP_STOPS[idx]
    c1 = _CMAP_STOPS[np.clip(idx + 1, 0, n)]
    return (c0 + (c1 - c0) * frac).astype(np.uint8)


def _region_label(cy: float, cx: float) -> str:
    """cy, cx in [0,1] -> a human-readable 3x3 grid location on the leaf."""
    row = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "middle")
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    if row == "middle" and col == "center":
        return "center of the leaf"
    return f"{row} {col}".replace("middle center", "center")


def run_gradcam(tensor: torch.Tensor, class_idx: int, base_img: Image.Image):
    """Returns (heatmap_data_uri, focus_region_label, coverage_fraction)."""
    model.zero_grad(set_to_none=True)
    tensor = tensor.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = model(tensor)
        score = logits[0, class_idx]
        score.backward()

    acts = _activations["value"][0]       # [C, h, w]
    grads = _gradients["value"][0]        # [C, h, w]
    weights = grads.mean(dim=(1, 2))      # [C]
    cam = F.relu((weights[:, None, None] * acts).sum(dim=0))  # [h, w]
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.detach().cpu().numpy()

    # focus region from the CAM's own (low-res) centroid, before upsampling
    ys, xs = np.mgrid[0:cam_np.shape[0], 0:cam_np.shape[1]]
    total = cam_np.sum() + 1e-8
    cy = float((ys * cam_np).sum() / total) / max(cam_np.shape[0] - 1, 1)
    cx = float((xs * cam_np).sum() / total) / max(cam_np.shape[1] - 1, 1)
    coverage = float((cam_np > 0.5).sum()) / cam_np.size

    # upsample CAM to image size and blend over the actual (resized) input image
    cam_t = torch.tensor(cam_np)[None, None]
    cam_up = F.interpolate(cam_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()
    heat_rgb = _colorize(cam_up)

    base_resized = base_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    base_np = np.array(base_resized).astype(np.float32)
    alpha = np.clip(cam_up, 0.25, 0.85)[..., None]  # keep low-activation areas mostly original image
    blended = (base_np * (1 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)

    out_img = Image.fromarray(blended)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return data_uri, _region_label(cy, cx), round(coverage, 3)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "classes": classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idx = probs.topk(min(3, len(classes)))
    top3 = [
        {"label": classes[i], "confidence": round(p.item(), 4)}
        for p, i in zip(top_probs, top_idx)
    ]
    best_label = top3[0]["label"]
    best_conf = top3[0]["confidence"]
    best_idx = int(top_idx[0].item())
    info = ADVICE.get(best_label, {"severity": "unknown", "summary": "", "action": "Consult a tea agronomist."})

    heatmap_uri, focus_region, coverage = None, None, None
    try:
        heatmap_uri, focus_region, coverage = run_gradcam(tensor, best_idx, image)
    except Exception as e:
        print(f"Grad-CAM failed (continuing without it): {e}")

    return {
        "label": best_label,
        "confidence": best_conf,
        "top3": top3,
        "severity": info["severity"],
        "summary": info["summary"],
        "action": info["action"],
        "heatmap": heatmap_uri,
        "focus_region": focus_region,
        "focus_coverage": coverage,
    }


if os.path.isdir(APP_DIR):
    app.mount("/app", StaticFiles(directory=APP_DIR, html=True), name="mobile_app")

    @app.get("/")
    def root():
        return FileResponse(os.path.join(APP_DIR, "index.html"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
