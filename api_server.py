# File: api_server.py
# Install dulu: pip install fastapi uvicorn python-multipart

import os
import json
import io
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torchvision.transforms as transforms

from core import config, model_builder

app = FastAPI(title="Aksara Jawa Classifier API", version="2.0")

# Mengizinkan akses dari Website (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bisa diganti dengan domain website spesifikmu nanti
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Global (Berjalan sekali saat server menyala)
device = torch.device("cpu")  # Server API umumnya pakai CPU agar hemat biaya hosting

with open(config.CLASS_MAP_PATH, 'r') as f:
    class_indices = json.load(f)
class_map = {int(v): k for k, v in class_indices.items()}
num_classes = len(class_map)

model = model_builder.build_model(num_classes)
model_path = getattr(config, 'MODEL_V2_SAVE_PATH', config.MODEL_SAVE_PATH)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.post("/predict")
async def predict_aksara(file: UploadFile = File(...)):
    try:
        # Baca gambar yang diunggah
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocessing & Prediksi
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        confidence, idx = torch.max(probabilities, dim=0)

        return {
            "status": "success",
            "aksara": class_map[idx.item()],
            "confidence": round(confidence.item() * 100, 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
