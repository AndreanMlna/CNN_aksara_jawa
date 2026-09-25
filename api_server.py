# File: api_server.py
"""
Backend REST API Server untuk CNN Aksara Jawa
Dibangun dengan FastAPI, PyTorch, dan standar arsitektur Computer Vision berkinerja tinggi.
"""

import os
import io
import time
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import torchvision.transforms as transforms

from core import config, model_builder
from core.aksara_data import get_aksara_details, SANDHANGAN_INFO

# Module Constants
DEFAULT_CONFIDENCE_THRESHOLD = 50.0
DEFAULT_INK_THRESHOLD = 190
CANVAS_HANDWRITING_FILENAME = "canvas_handwriting.png"

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="ꦄꦏ꧀ꦱꦫꦗꦮ · Aksara Jawa AI Vision Studio",
    description="Sistem Pengenalan Citra Aksara Jawa End-to-End berbasis EfficientNet-B0 (120 Kelas)",
    version="2.0.0"
)

# Konfigurasi Kebijakan CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Inisialisasi Model & Resource Global (In-Memory Engine)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Inisialisasi Model Engine pada Device: {device}")

# Muat Pemetaan Kelas
if not os.path.exists(config.CLASS_MAP_PATH):
    raise FileNotFoundError(f"File index kelas {config.CLASS_MAP_PATH} tidak ditemukan.")

with open(config.CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
    class_indices = json.load(f)

idx_to_class = {int(v): k for k, v in class_indices.items()}
class_to_idx = {v: k for k, v in idx_to_class.items()}
num_classes = len(class_indices)

# Inisialisasi Arsitektur EfficientNet-B0
model = model_builder.build_model(num_classes)
model_path = getattr(config, 'MODEL_V2_SAVE_PATH', config.MODEL_SAVE_PATH)
if not os.path.exists(model_path):
    model_path = config.MODEL_SAVE_PATH

print(f"[*] Memuat bobot model dari: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Pipeline Pra-pemrosesan Citra (Paritas Identik dengan Training & TFLite)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


@dataclass
class InkBoundingBox:
    """Representasi koordinat dan batas wilayah tinta goresan aksara."""
    arr: np.ndarray
    ink_y: np.ndarray
    ink_x: np.ndarray
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    sw: int
    sh: int


def extract_ink_bounding_box(
    raw_img: Image.Image,
    threshold: int = DEFAULT_INK_THRESHOLD,
    margin: int = 5
) -> Optional[InkBoundingBox]:
    """Mengekstrak matriks piksel tinta dan batas bounding box dengan perlindungan perimeter."""
    arr = np.array(raw_img.convert('L'))
    if arr.shape[0] > 30 and arr.shape[1] > 30:
        inner = arr[margin:-margin, margin:-margin]
        _, inner_ink_x = np.where(inner < threshold)
        if len(inner_ink_x) >= 20:
            arr = inner

    ink_y, ink_x = np.where(arr < threshold)
    if len(ink_x) < 25:
        return None

    min_x, max_x = int(ink_x.min()), int(ink_x.max())
    min_y, max_y = int(ink_y.min()), int(ink_y.max())

    return InkBoundingBox(
        arr=arr,
        ink_y=ink_y,
        ink_x=ink_x,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        sw=max_x - min_x + 1,
        sh=max_y - min_y + 1
    )


def detect_suku_descender(raw_img: Image.Image) -> tuple[bool, float]:
    """Mendeteksi keberadaan fisik sandhangan suku (ekor vertikal ke bawah di kuadran kanan)."""
    bbox = extract_ink_bounding_box(raw_img)
    if bbox is None:
        return False, 0.0

    x_cutoff = bbox.min_x + int(0.55 * bbox.sw)
    body_ink_y = bbox.ink_y[bbox.ink_x <= x_cutoff]
    tail_ink_y = bbox.ink_y[bbox.ink_x > x_cutoff]

    if len(body_ink_y) == 0 or len(tail_ink_y) == 0:
        return False, 0.0

    body_base = int(body_ink_y.max())
    body_top = int(body_ink_y.min())
    body_h = max(1, body_base - body_top + 1)
    tail_bot = int(tail_ink_y.max())

    descender_px = tail_bot - body_base
    ratio = descender_px / body_h

    has_suku = (descender_px >= 35) and (ratio >= 0.35)
    return has_suku, max(0.0, ratio)


def detect_taling_and_tarung(raw_img: Image.Image) -> tuple[bool, bool]:
    """Deteksi Morfologis Struktur Sandhangan Taling (2 glif) vs Taling-Tarung (3 glif)."""
    bbox = extract_ink_bounding_box(raw_img)
    if bbox is None:
        return False, False

    aspect = bbox.sw / max(bbox.sh, 1)
    crop_ink = (bbox.arr[bbox.min_y:bbox.max_y + 1, bbox.min_x:bbox.max_x + 1] < DEFAULT_INK_THRESHOLD).astype(np.uint8)

    has_taling = False
    if aspect >= 0.95:
        sep_start = max(1, int(bbox.sw * 0.20))
        sep_end = min(bbox.sw - 1, int(bbox.sw * 0.48))
        if sep_end > sep_start:
            proj_l = crop_ink[:, sep_start:sep_end].sum(axis=0)
            if len(proj_l) > 0 and proj_l.min() <= 2:
                valley_x = sep_start + int(np.argmin(proj_l))
                left_col_sums = crop_ink[:, :valley_x].sum(axis=1)
                left_h = (left_col_sums > 0).sum()
                if left_h >= 0.55 * bbox.sh:
                    has_taling = True

    has_tarung = False
    if aspect >= 1.25:
        proj_r = crop_ink[:, int(bbox.sw * 0.55):int(bbox.sw * 0.88)].sum(axis=0)
        if len(proj_r) > 0 and proj_r.min() <= 3:
            far_right_ink = crop_ink[:, int(bbox.sw * 0.85):].sum()
            if far_right_ink >= 25:
                has_tarung = True

    return has_taling, has_tarung


def normalize_canvas_stroke(raw_img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Ekstraksi Bounding-Box & Normalisasi Aspek Rasio Goresan Tulisan Tangan Kanvas."""
    arr = np.array(raw_img.convert('L'))
    offset_x, offset_y = 0, 0
    if arr.shape[0] > 20 and arr.shape[1] > 20:
        inner = arr[3:-3, 3:-3]
        _, ink_x_in = np.where(inner < DEFAULT_INK_THRESHOLD)
        if len(ink_x_in) >= 20:
            arr = inner
            offset_x, offset_y = 3, 3

    ink_y, ink_x = np.where(arr < DEFAULT_INK_THRESHOLD)
    if len(ink_x) < 25:
        return raw_img, raw_img

    min_x, max_x = int(ink_x.min()) + offset_x, int(ink_x.max()) + offset_x
    min_y, max_y = int(ink_y.min()) + offset_y, int(ink_y.max()) + offset_y
    sw = max_x - min_x + 1
    sh = max_y - min_y + 1
    stroke = raw_img.crop((min_x, min_y, max_x + 1, max_y + 1))

    # View 0: Native bujur sangkar (500x500)
    v0_img = Image.new('RGB', (500, 500), (255, 255, 255))
    scale_0 = min(360 / max(sw, 1), 360 / max(sh, 1))
    nw0 = max(1, int(sw * scale_0))
    nh0 = max(1, int(sh * scale_0))
    r_stroke_0 = stroke.resize((nw0, nh0), Image.Resampling.LANCZOS)
    v0_img.paste(r_stroke_0, ((500 - nw0) // 2, (500 - nh0) // 2))

    # View Card: Kartu putih 600x500
    card_img = Image.new('RGB', (600, 500), (255, 255, 255))
    scale_c = min(360 / max(sw, 1), 340 / max(sh, 1))
    nwc = max(1, int(sw * scale_c))
    nhc = max(1, int(sh * scale_c))

    # Adaptive width relaxation: Cegah kolaps punuk ganda (misal Ga -> Gu) saat goresan memiliki ekor suku panjang
    if nwc < 250 and sw / max(sh, 1) < 0.85:
        nwc = min(360, max(nwc, min(255, int(nwc * 1.25))))

    r_stroke_c = stroke.resize((nwc, nhc), Image.Resampling.LANCZOS)

    px = max(15, min(600 - nwc - 15, 245 - nwc // 2))
    py = max(15, min(500 - nhc - 10, 430 - nhc))
    card_img.paste(r_stroke_c, (px, py))

    return v0_img, card_img


def generate_multiview_canvas_tensors(card_norm: Image.Image, v0_norm: Image.Image) -> torch.Tensor:
    """Menghasilkan 4-view tensor batch untuk kanvas tulis digital."""
    v0 = image_transform(v0_norm)

    t1 = Image.new('RGB', (1528, 540), (0, 0, 0))
    t1.paste(card_norm, (464, 20))
    v1 = image_transform(t1)

    t2 = Image.new('RGB', (882, 540), (0, 0, 0))
    t2.paste(card_norm, (140, 20))
    v2 = image_transform(t2)

    t3 = Image.new('RGB', (683, 540), (0, 0, 0))
    t3.paste(card_norm, (41, 20))
    v3 = image_transform(t3)

    return torch.stack([v0, v1, v2, v3]).to(device)


def blend_canvas_domain_probabilities(
    probs: torch.Tensor,
    has_suku: bool,
    has_taling: bool,
    has_tarung: bool
) -> torch.Tensor:
    """Menggabungkan konsensus multi-view probabilistik berdasarkan bukti morfologi kanvas."""
    p_square = probs[0]
    p_ultra_wide = probs[1]
    p_medium_wide = probs[2]
    p_taling = probs[3]

    final_probs = torch.zeros_like(p_square)

    if has_suku:
        # Dataset-grounded: 16 kelas sandhangan suku dengan rasio ultra-wide (1528x540 / 1512x540)
        wide_suku_keys = [
            "bu", "dhu", "du", "gu", "hu", "ju", "lu", "mu",
            "ngu", "nyu", "pu", "su", "thu", "tu", "wu", "yu"
        ]
        wide_suku_score = sum(
            p_ultra_wide[class_to_idx[f"suku_{k}"]].item()
            for k in wide_suku_keys if f"suku_{k}" in class_to_idx
        )
        is_wide_dom = (wide_suku_score >= 0.20)

        # Sinergi Bayesian konsonan dasar dari View 0 (Square)
        consonants_map = {
            'ga': 'gu', 'ra': 'ru', 'ka': 'ku', 'ca': 'cu', 'ba': 'bu',
            'ja': 'ju', 'da': 'du', 'sa': 'su', 'ta': 'tu', 'na': 'nu',
            'pa': 'pu', 'la': 'lu', 'ma': 'mu', 'wa': 'wu', 'ya': 'yu',
            'ha': 'hu', 'dha': 'dhu', 'tha': 'thu', 'nga': 'ngu', 'nya': 'nyu'
        }
        base_evidence = {}
        for base_c, suku_c in consonants_map.items():
            b_cls = f"aksara-dasar_{base_c}"
            if b_cls in class_to_idx:
                base_evidence[f"suku_{suku_c}"] = p_square[class_to_idx[b_cls]].item()

        for idx in range(num_classes):
            c_name = idx_to_class[idx]
            if c_name.startswith("suku_"):
                base_boost = base_evidence.get(c_name, 0.0)
                if is_wide_dom:
                    final_probs[idx] = 0.70 * p_ultra_wide[idx] + 0.15 * p_medium_wide[idx] + 0.15 * (p_square[idx] + base_boost)
                else:
                    final_probs[idx] = 0.40 * p_ultra_wide[idx] + 0.40 * p_medium_wide[idx] + 0.20 * (p_square[idx] + base_boost)
            elif c_name.startswith("aksara-dasar_") or c_name.startswith("pepet_"):
                final_probs[idx] = 0.20 * p_square[idx]
            elif c_name.startswith("wulu_") or c_name.startswith("taling-tarung_") or c_name.startswith("taling_"):
                final_probs[idx] = 0.25 * p_medium_wide[idx]
            else:
                final_probs[idx] = 0.10 * p_medium_wide[idx]

    elif has_taling and not has_tarung:
        for idx in range(num_classes):
            cat = idx_to_class[idx].split('_')[0]
            if cat == 'taling':
                final_probs[idx] = 0.65 * p_taling[idx] + 0.35 * p_medium_wide[idx]
            elif cat in ['aksara-dasar', 'pepet']:
                final_probs[idx] = 0.25 * p_square[idx]
            else:
                final_probs[idx] = 0.10 * p_medium_wide[idx]

    elif has_taling and has_tarung:
        for idx in range(num_classes):
            cat = idx_to_class[idx].split('_')[0]
            if cat == 'taling-tarung':
                final_probs[idx] = 0.70 * p_ultra_wide[idx] + 0.30 * p_medium_wide[idx]
            elif cat in ['aksara-dasar', 'pepet']:
                final_probs[idx] = 0.15 * p_square[idx]
            else:
                final_probs[idx] = 0.10 * p_medium_wide[idx]

    else:
        for idx in range(num_classes):
            cat = idx_to_class[idx].split('_')[0]
            if cat in ['aksara-dasar', 'pepet']:
                final_probs[idx] = 0.80 * p_square[idx] + 0.20 * p_medium_wide[idx]
            elif cat == 'wulu':
                final_probs[idx] = 0.75 * p_medium_wide[idx] + 0.25 * p_square[idx]
            else:
                final_probs[idx] = 0.15 * p_square[idx]

    p_sum = torch.sum(final_probs)
    return final_probs / p_sum if p_sum > 0 else final_probs


def infer_probabilities(image_bytes: bytes, is_canvas: bool = False) -> torch.Tensor:
    """Domain-Aware Hierarchical Consensus Fusion Inference Engine."""
    raw_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Jalur 1: Citra unggahan asli (100% Paritas Murni dengan Training, Evaluasi & TFLite)
    if not is_canvas:
        tensor = image_transform(raw_img).unsqueeze(0).to(device)
        with torch.no_grad():
            return F.softmax(model(tensor), dim=1)[0]

    # Jalur 2: Citra kanvas tulis web
    has_suku, _ = detect_suku_descender(raw_img)
    has_taling, has_tarung = detect_taling_and_tarung(raw_img)
    v0_norm, card_norm = normalize_canvas_stroke(raw_img)

    batch = generate_multiview_canvas_tensors(card_norm, v0_norm)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)

    return blend_canvas_domain_probabilities(probs, has_suku, has_taling, has_tarung)


def format_top_predictions(probabilities: torch.Tensor, top_k: int = 5) -> list[dict]:
    """Mengonversi tensor probabilitas menjadi daftar kandidat teratas beserta metadata."""
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, num_classes))
    top_results = []
    for prob, idx in zip(top_probs, top_indices):
        c_name = idx_to_class[idx.item()]
        c_details = get_aksara_details(c_name)
        top_results.append({
            "class_name": c_name,
            "confidence": round(prob.item() * 100, 2),
            "unicode_char": c_details["unicode_char"],
            "latin": c_details["latin"],
            "category": c_details["category"],
            "category_name": c_details["category_name"],
            "desc": c_details["desc"],
            "base_consonant": c_details.get("base_consonant", "Baku"),
            "vowel": c_details.get("vowel", "a"),
            "position": c_details.get("position", "Bentuk Baku")
        })
    return top_results


def evaluate_quiz_match(
    predicted_name: str,
    target_class: str,
    predicted_conf: float
) -> tuple[str, bool, bool, str]:
    """Mengevaluasi kesesuaian antara aksara prediksi model dan target kuis."""
    pred_details = get_aksara_details(predicted_name)
    target_details = get_aksara_details(target_class)

    is_exact = (predicted_name == target_class)
    is_close = False

    if is_exact:
        if predicted_conf >= 75.0:
            verdict = "EXACT"
            feedback = "Luar biasa! Bentuk dan proporsi aksara yang Anda tulis sangat tepat dan jelas."
        else:
            verdict = "GOOD"
            feedback = "Aksara sudah tepat, namun goresan dapat dipertegas agar tingkat kepastian model lebih tinggi."
    else:
        if pred_details.get("base_consonant") == target_details.get("base_consonant"):
            is_close = True
            verdict = "SANDHANGAN_MISMATCH"
            feedback = (f"Aksara dasar sudah benar ({target_details['base_consonant']}), "
                        f"namun sandhangan tertukar dengan {pred_details['category_name']}.")
        else:
            verdict = "INCORRECT"
            feedback = (f"Kurang tepat. Model mendeteksi aksara '{pred_details['latin']}' "
                        f"({pred_details['unicode_char']}), sedangkan target yang diminta adalah '{target_details['latin']}' ({target_details['unicode_char']}).")

    return verdict, is_exact, is_close, feedback


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.post("/api/predict")
async def predict_aksara(file: UploadFile = File(...)):
    """
    Endpoint utama untuk klasifikasi citra Aksara Jawa.
    Menerima file citra (JPG, PNG, WebP), mengembalikan Top-1 & Top-5 kandidat beserta metadata.
    """
    start_time = time.perf_counter()
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="File citra kosong.")

        is_canvas = (file.filename == CANVAS_HANDWRITING_FILENAME)
        probabilities = infer_probabilities(image_bytes, is_canvas=is_canvas)
        top5_results = format_top_predictions(probabilities, top_k=5)

        best_result = top5_results[0]
        top1_conf = best_result.get("confidence", 0.0)
        is_confident = bool(top1_conf >= DEFAULT_CONFIDENCE_THRESHOLD)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "status": "success",
            "is_confident": is_confident,
            "threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "clarification_message": None if is_confident else (
                f"Tulisan kurang jelas (Tingkat Keyakinan {top1_conf}% < {DEFAULT_CONFIDENCE_THRESHOLD}%). "
                f"Apakah yang kamu maksud salah satu dari 5 kemungkinan teratas ini?"
            ),
            "predicted": best_result,
            "top5": top5_results,
            "latency_ms": elapsed_ms,
            "device": str(device),
            "model_architecture": "EfficientNet-B0",
            "model_version": "V2 (Fine-Tuned)"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/verify")
async def verify_aksara(
    file: UploadFile = File(...),
    target_class: str = Form(...)
):
    """
    Endpoint evaluasi kuis / mode latihan menulis.
    Memeriksa kecocokan antara gambar tulisan tangan dengan target aksara yang diminta.
    """
    start_time = time.perf_counter()
    try:
        image_bytes = await file.read()
        probabilities = infer_probabilities(image_bytes, is_canvas=True)

        top_prob, top_idx = torch.max(probabilities, dim=0)
        predicted_name = idx_to_class[top_idx.item()]
        predicted_conf = round(top_prob.item() * 100, 2)

        verdict, is_exact, is_close, feedback = evaluate_quiz_match(
            predicted_name, target_class, predicted_conf
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "status": "success",
            "verdict": verdict,
            "is_correct": is_exact,
            "is_close": is_close,
            "confidence": predicted_conf,
            "feedback": feedback,
            "predicted": get_aksara_details(predicted_name),
            "target": get_aksara_details(target_class),
            "latency_ms": elapsed_ms
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/classes")
async def get_all_classes():
    """Mengembalikan seluruh daftar 120 kelas yang dikelompokkan berdasarkan kategori sandhangan."""
    categorized = {}
    for c_name in sorted(class_indices.keys()):
        details = get_aksara_details(c_name)
        details["index"] = class_indices[c_name]
        cat = details["category"]
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(details)

    return {
        "status": "success",
        "total_classes": num_classes,
        "categories_info": SANDHANGAN_INFO,
        "data": categorized
    }


@app.get("/api/info")
async def get_system_info():
    """Mengembalikan spesifikasi sistem, parameter model, dan ringkasan metrik performa."""
    return {
        "status": "success",
        "model_name": "CNN Aksara Jawa Classifier",
        "architecture": "EfficientNet-B0 (Compound Scaling)",
        "classifier_head": "1280 -> Dense(512, BN, ReLU, Drop 0.4) -> Dense(256, BN, ReLU, Drop 0.3) -> Dense(120, Drop 0.2)",
        "input_resolution": "224x224x3 (RGB)",
        "normalization": "ImageNet (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])",
        "total_classes": num_classes,
        "validation_accuracy": "98.30%",
        "tflite_parity_accuracy": "98.30%",
        "quantization_strategy": "Pure Float32 (Zero Quantization Degradation)",
        "device": str(device),
        "pytorch_version": torch.__version__,
        "export_targets": ["Android TFLite (.tflite)", "PyTorch Mobile (.ptl)", "Web ONNX (.onnx)", "FastAPI REST API"]
    }


@app.get("/api/sample-images")
async def get_sample_images():
    """Mengembalikan daftar nama kelas populer yang dapat dicoba sebagai sampel gambar cepat."""
    samples = [
        {"class_name": "aksara-dasar_ba", "latin": "Ba", "char": "ꦧ", "category": "Aksara Dasar"},
        {"class_name": "aksara-dasar_ha", "latin": "Ha", "char": "ꦲ", "category": "Aksara Dasar"},
        {"class_name": "aksara-dasar_ka", "latin": "Ka", "char": "ꦏ", "category": "Aksara Dasar"},
        {"class_name": "wulu_hi", "latin": "Hi", "char": "ꦲꦶ", "category": "Wulu (i)"},
        {"class_name": "suku_du", "latin": "Du", "char": "ꦢꦸ", "category": "Suku (u)"},
        {"class_name": "taling_nge", "latin": "Ngé", "char": "ꦔꦺ", "category": "Taling (é)"},
        {"class_name": "pepet_yê", "latin": "Yê", "char": "ꦪꦼ", "category": "Pepet (ê)"},
        {"class_name": "taling-tarung_bo", "latin": "Bo", "char": "ꦧꦺꦴ", "category": "Taling-Tarung (o)"}
    ]
    return {"status": "success", "samples": samples}


# ---------------------------------------------------------
# Static Files & Web Frontend Mounting
# ---------------------------------------------------------
if os.path.exists("web"):
    app.mount("/static", StaticFiles(directory="web"), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse("web/index.html")
