"""
FastAPI inference server for the trained tea-leaf disease classifier.

Loads checkpoints/best_model.pt (MobileNetV2, 9 classes) and exposes:
  POST /predict  - multipart image upload -> JSON {label, confidence, top3, advice}
  GET  /health   - liveness check
  GET  /         - serves the mobile web app (mobile_app/index.html)

Run:  python3 server.py
Then open http://<this-machine's-LAN-IP>:8000 on your phone (same Wi-Fi).
"""
import io
import json
import os

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

app = FastAPI(title="Tea Leaf Health Scanner API")
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
    info = ADVICE.get(best_label, {"severity": "unknown", "summary": "", "action": "Consult a tea agronomist."})

    return {
        "label": best_label,
        "confidence": best_conf,
        "top3": top3,
        "severity": info["severity"],
        "summary": info["summary"],
        "action": info["action"],
    }


if os.path.isdir(APP_DIR):
    app.mount("/app", StaticFiles(directory=APP_DIR, html=True), name="mobile_app")

    @app.get("/")
    def root():
        return FileResponse(os.path.join(APP_DIR, "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
