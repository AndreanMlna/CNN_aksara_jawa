# File: api_server.py
"""
Backend REST API Server untuk CNN Aksara Jawa
Dibangun dengan FastAPI, PyTorch, dan standar arsitektur Computer Vision berkinerja tinggi.
"""

import os
import io
import time
import json
import glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import torchvision.transforms as transforms

from core import config, model_builder
from core.aksara_data import get_aksara_details, BASE_CONSONANTS, SANDHANGAN_INFO

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
num_classes = len(class_indices)

# Kelompokkan indeks kelas berdasarkan domain kategori dataset
cat_to_indices = {c: [] for c in ['aksara-dasar', 'pepet', 'suku', 'taling-tarung', 'wulu', 'taling']}
for name, idx in class_indices.items():
    cat = name.split('_')[0]
    if cat in cat_to_indices:
        cat_to_indices[cat].append(int(idx))

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


class_to_idx = {v: k for k, v in idx_to_class.items()}


def detect_suku_descender(raw_img: Image.Image) -> tuple[bool, float]:
    """
    Mendeteksi keberadaan fisik sandhangan suku (ekor vertikal ke bawah di kuadran kanan).
    Algoritma Computer Vision berbasis analisis morfologis proyeksi adaptif:
    - Ambang pemisah badan dan ekor ditetapkan pada 55% lebar aksara (sw).
    - Badan aksara (0% - 55% lebar): mencakup seluruh lengkungan punuk & baseline badan aksara.
    - Ekor sandhangan suku (55% - 100% lebar): menjulur jauh ke bawah melampaui garis dasar badan aksara.
    - Mencegah false positive pada aksara dasar ramping (seperti Ra dan Ga) yang kaki kirinya
      terkadang digambar lebih pendek/menggantung oleh pengguna.
    - Return: (has_suku: bool, descender_ratio: float)
    """
    arr = np.array(raw_img.convert('L'))
    # Guard terhadap noise border HTML canvas / screenshot artifacts (1-5px perimeter)
    if arr.shape[0] > 30 and arr.shape[1] > 30:
        inner = arr[5:-5, 5:-5]
        ink_y_in, ink_x_in = np.where(inner < 190)
        if len(ink_x_in) >= 20:
            arr = inner

    ink_y, ink_x = np.where(arr < 190)
    if len(ink_x) < 25:
        return False, 0.0

    min_x, max_x = int(ink_x.min()), int(ink_x.max())
    min_y, max_y = int(ink_y.min()), int(ink_y.max())
    sw = max_x - min_x + 1
    sh = max_y - min_y + 1

    # Ambang batas pemisah badan (kiri 55%) vs ekor suku (kanan 45%)
    x_cutoff = min_x + int(0.55 * sw)
    body_ink_y = ink_y[ink_x <= x_cutoff]
    tail_ink_y = ink_y[ink_x > x_cutoff]

    if len(body_ink_y) == 0 or len(tail_ink_y) == 0:
        return False, 0.0

    body_base = int(body_ink_y.max())
    body_top = int(body_ink_y.min())
    b_h = max(1, body_base - body_top + 1)
    tail_bot = int(tail_ink_y.max())

    descender_px = tail_bot - body_base
    ratio = descender_px / b_h

    # Sandhangan suku sejati: ekor turun minimal 20px di bawah dasar badan dan rasio >= 0.25
    has_suku = (descender_px >= 20) and (ratio >= 0.25)
    return has_suku, max(0.0, ratio)


def detect_taling_and_tarung(raw_img: Image.Image):
    """
    Deteksi Morfologis Struktur Sandhangan Taling (2 glif) vs Taling-Tarung (3 glif).
    - Taling (ꦺ): Glif sandhangan di sisi kiri (x <= 0.48 * sw) dengan kaki vertikal ke bawah,
      diikuti aksara konsonan di sisi kanan.
    - Taling-Tarung (ꦺ...ꦴ): Format 3 glif (Taling kiri, Konsonan tengah, Tarung kanan)
      dengan rasio aspek lebar (sw / sh >= 1.25) dan adanya lembah spasi pemisah sebelum Tarung.
    """
    arr = np.array(raw_img.convert('L'))
    if arr.shape[0] > 30 and arr.shape[1] > 30:
        inner = arr[5:-5, 5:-5]
        ink_y_in, ink_x_in = np.where(inner < 190)
        if len(ink_x_in) >= 20:
            arr = inner

    ink_y, ink_x = np.where(arr < 190)
    if len(ink_x) < 25:
        return False, False

    min_x, max_x = int(ink_x.min()), int(ink_x.max())
    min_y, max_y = int(ink_y.min()), int(ink_y.max())
    sw = max_x - min_x + 1
    sh = max_y - min_y + 1

    aspect = sw / max(sh, 1)
    if aspect < 0.60:
        return False, False

    # Analisis glif kiri (kandidat Taling)
    left_mask = (ink_x >= min_x) & (ink_x <= min_x + int(0.48 * sw))
    left_ink_y = ink_y[left_mask]
    left_ink_x = ink_x[left_mask]

    right_mask = (ink_x > min_x + int(0.48 * sw))
    right_ink_y = ink_y[right_mask]
    right_ink_x = ink_x[right_mask]

    if len(left_ink_x) < 15 or len(right_ink_x) < 15:
        return False, False

    left_h = int(left_ink_y.max()) - int(left_ink_y.min()) + 1
    # Taling memiliki tinggi vertikal signifikan (>= 60% dari total bounding box tinggi)
    has_taling = (left_h >= 0.60 * sh)

    # Deteksi komponen Tarung di sisi kanan (Aksara 3 glif)
    has_tarung = False
    if aspect >= 1.25:
        crop_ink = (arr[min_y:max_y + 1, min_x:max_x + 1] < 190).astype(np.uint8)
        # Lembah pemisah antara konsonan tengah dan tarung kanan pada rentang 55% - 88% lebar
        proj_r = crop_ink[:, int(sw * 0.55):int(sw * 0.88)].sum(axis=0)
        if len(proj_r) > 0 and proj_r.min() <= 3:
            far_right_ink = crop_ink[:, int(sw * 0.85):].sum()
            if far_right_ink >= 25:
                has_tarung = True

    return has_taling, has_tarung


def normalize_canvas_stroke(raw_img: Image.Image):
    """
    Ekstraksi Bounding-Box & Normalisasi Aspek Rasio Goresan Tulisan Tangan Kanvas.
    Menyelaraskan skala spasial, margin, dan posisi baseline goresan kanvas web
    agar 100% kongruen dengan distribusi spasial dataset EfficientNet-B0.
    """
    arr = np.array(raw_img.convert('L'))
    # Guard terhadap perimeter border
    offset_x, offset_y = 0, 0
    if arr.shape[0] > 20 and arr.shape[1] > 20:
        inner = arr[3:-3, 3:-3]
        ink_y_in, ink_x_in = np.where(inner < 190)
        if len(ink_x_in) >= 20:
            arr = inner
            offset_x, offset_y = 3, 3

    ink_y, ink_x = np.where(arr < 190)
    if len(ink_x) < 25:
        # Kanvas kosong
        return raw_img, raw_img

    min_x, max_x = int(ink_x.min()) + offset_x, int(ink_x.max()) + offset_x
    min_y, max_y = int(ink_y.min()) + offset_y, int(ink_y.max()) + offset_y
    sw = max_x - min_x + 1
    sh = max_y - min_y + 1
    stroke = raw_img.crop((min_x, min_y, max_x + 1, max_y + 1))

    # 1. View 0: Domain Bujur Sangkar (Native untuk Aksara Dasar & Pepet: 500x500 dengan margin natural)
    v0_img = Image.new('RGB', (500, 500), (255, 255, 255))
    scale_0 = min(360 / max(sw, 1), 360 / max(sh, 1))
    nw0 = max(1, int(sw * scale_0))
    nh0 = max(1, int(sh * scale_0))
    r_stroke_0 = stroke.resize((nw0, nh0), Image.Resampling.LANCZOS)
    v0_img.paste(r_stroke_0, ((500 - nw0) // 2, (500 - nh0) // 2))

    # 2. Card Domain: Kartu Putih Bersih 600x500 (Native untuk Suku, Wulu, Taling, Taling-Tarung)
    card_img = Image.new('RGB', (600, 500), (255, 255, 255))
    scale_c = min(360 / max(sw, 1), 340 / max(sh, 1))
    nwc = max(1, int(sw * scale_c))
    nhc = max(1, int(sh * scale_c))
    r_stroke_c = stroke.resize((nwc, nhc), Image.Resampling.LANCZOS)

    # Baseline anchoring: Kongruen dengan distribusi tinggi & baseline dataset asli (y ~ 430)
    px = max(15, min(600 - nwc - 15, 245 - nwc // 2))
    py = max(15, min(500 - nhc - 10, 430 - nhc))
    card_img.paste(r_stroke_c, (px, py))

    return v0_img, card_img


def infer_probabilities(image_bytes: bytes) -> torch.Tensor:
    """
    Domain-Aware Hierarchical Consensus Fusion Inference Engine.
    Menyelaraskan disparitas spasial & template antara berkas dataset asli dengan kanvas web:
    - Jika citra berasal dari berkas unggahan: langsung diproses dengan standar paritas murni.
    - Jika citra berasal dari kanvas digital (latar putih solid):
      * Deteksi morfologis ekor sandhangan suku (kuadran kanan bawah).
      * Deteksi morfologis glif sandhangan taling kiri dan tarung kanan.
      * Routing domain-aware:
        - Suku terdeteksi: View 1 (ultra-wide) untuk suku lebar, View 2 untuk suku medium.
        - Taling murni (tanpa tarung): View 3 (683x540 native Taling) + View 2. Supresi Taling-Tarung palsu.
        - Taling-Tarung (3 glif): View 1 (1528x540) + View 2.
        - Aksara-dasar / Pepet / Wulu: View 0 (White 1:1) + View 2.
    """
    raw_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    w, h = raw_img.size
    aspect = w / max(h, 1)

    arr_l = np.array(raw_img.convert('L'))
    p = min(5, min(w, h))
    corners = [
        arr_l[:p, :p].mean(),
        arr_l[:p, -p:].mean(),
        arr_l[-p:, :p].mean(),
        arr_l[-p:, -p:].mean()
    ]
    is_white_canvas = (0.7 <= aspect <= 1.4) and (float(np.mean(corners)) > 90.0)

    # Jalur 1: Citra unggahan asli (di luar kanvas bujur sangkar putih)
    if not is_white_canvas:
        t = image_transform(raw_img).unsqueeze(0).to(device)
        with torch.no_grad():
            return F.softmax(model(t), dim=1)[0]

    # Jalur 2: Citra kanvas tulis web
    has_suku, _ = detect_suku_descender(raw_img)
    has_taling, has_tarung = detect_taling_and_tarung(raw_img)
    v0_norm, card_norm = normalize_canvas_stroke(raw_img)

    # View 0: Native domain aksara-dasar & pepet (Bujur Sangkar Putih 500x500)
    v0 = image_transform(v0_norm)

    # View 1: Native domain suku lebar (Du, Su, Bu, Dhu, Lu, Mu, Pu, dll) & taling-tarung (1528x540)
    t1 = Image.new('RGB', (1528, 540), (0, 0, 0))
    t1.paste(card_norm, (464, 20))
    v1 = image_transform(t1)

    # View 2: Native domain suku medium (Cu: 882x540) serta Wulu
    t2 = Image.new('RGB', (882, 540), (0, 0, 0))
    t2.paste(card_norm, (140, 20))
    v2 = image_transform(t2)

    # View 3: Native domain Taling 2 glif (683x540)
    t3 = Image.new('RGB', (683, 540), (0, 0, 0))
    t3.paste(card_norm, (41, 20))
    v3 = image_transform(t3)

    batch = torch.stack([v0, v1, v2, v3]).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)  # shape: (4, num_classes)

    p_v0 = probs[0]  # White 1:1
    p_v1 = probs[1]  # 1528x540 Ultra-wide
    p_v2 = probs[2]  # 882x540 Medium-wide
    p_v3 = probs[3]  # 683x540 Native Taling

    final_probs = torch.zeros_like(p_v0)

    if has_suku:
        # Sandhangan Suku terdeteksi secara fisik via morfologi ekor kuadran kanan bawah
        # Supresi aksara-dasar & pepet ke 0.0
        wide_suku_keys = ["su", "du", "pu", "bu", "dhu", "ju"]
        wide_suku_score = sum(p_v1[class_to_idx[f"suku_{k}"]].item() for k in wide_suku_keys if f"suku_{k}" in class_to_idx)

        if wide_suku_score > 0.20:
            # Karakter rumpun multi-punuk lebar (Su, Du, Pu, Ju, Bu, Dhu): Gunakan View 1 secara eksklusif, supresi Cu View 2
            for idx in range(num_classes):
                c_name = idx_to_class[idx]
                if c_name.startswith("suku_"):
                    if c_name == "suku_cu":
                        final_probs[idx] = p_v1[idx] * 0.01
                    else:
                        final_probs[idx] = p_v1[idx]
        else:
            # Karakter rumpun sempit/medium (Cu, Ku, Nu, Ru, Hu): View 2 primer
            for idx in range(num_classes):
                c_name = idx_to_class[idx]
                if c_name.startswith("suku_"):
                    suku_type = c_name.split("_")[1]
                    if suku_type in ["cu", "ku", "nu", "ru", "hu"]:
                        final_probs[idx] = 0.85 * p_v2[idx] + 0.15 * p_v1[idx]
                    else:
                        final_probs[idx] = p_v1[idx]

    elif has_taling and not has_tarung:
        # Sandhangan Taling Murni (2 glif: taling kiri + konsonan kanan, tanpa tarung)
        # Menolak proyeksi palsu Go / Po dari template 1528x540
        for idx in range(num_classes):
            c_name = idx_to_class[idx]
            cat = c_name.split('_')[0]
            if cat == 'taling':
                final_probs[idx] = 0.55 * p_v3[idx] + 0.45 * p_v2[idx]

    elif has_taling and has_tarung:
        # Sandhangan Taling-Tarung (3 glif: taling kiri + konsonan tengah + tarung kanan)
        for idx in range(num_classes):
            c_name = idx_to_class[idx]
            cat = c_name.split('_')[0]
            if cat == 'taling-tarung':
                final_probs[idx] = 0.70 * p_v1[idx] + 0.30 * p_v2[idx]

    else:
        # Karakter Aksara Dasar (Nglegena), Pepet, atau Wulu
        for idx in range(num_classes):
            c_name = idx_to_class[idx]
            cat = c_name.split('_')[0]
            if cat in ['aksara-dasar', 'pepet']:
                final_probs[idx] = 0.75 * p_v0[idx] + 0.25 * p_v2[idx]
            elif cat == 'wulu':
                final_probs[idx] = 0.75 * p_v2[idx] + 0.25 * p_v0[idx]

    # Normalisasi kembali probabilitas agar sum = 1
    final_probs = final_probs / torch.sum(final_probs)
    return final_probs


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

        probabilities = infer_probabilities(image_bytes)

        # Ambil Top 5 prediksi
        top5_probs, top5_indices = torch.topk(probabilities, k=min(5, num_classes))

        top5_results = []
        for prob, idx in zip(top5_probs, top5_indices):
            c_name = idx_to_class[idx.item()]
            c_details = get_aksara_details(c_name)
            top5_results.append({
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

        best_result = top5_results[0]
        top1_conf = best_result.get("confidence", 0.0)
        CONFIDENCE_THRESHOLD = 50.0
        is_confident = bool(top1_conf >= CONFIDENCE_THRESHOLD)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "status": "success",
            "is_confident": is_confident,
            "threshold": CONFIDENCE_THRESHOLD,
            "clarification_message": None if is_confident else f"Tulisan kurang jelas (Tingkat Keyakinan {top1_conf}% < {CONFIDENCE_THRESHOLD}%). Apakah yang kamu maksud salah satu dari 5 kemungkinan teratas ini?",
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
        probabilities = infer_probabilities(image_bytes)

        top_prob, top_idx = torch.max(probabilities, dim=0)
        predicted_name = idx_to_class[top_idx.item()]
        predicted_conf = round(top_prob.item() * 100, 2)

        pred_details = get_aksara_details(predicted_name)
        target_details = get_aksara_details(target_class)

        # Evaluasi kecocokan
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
            # Periksa apakah konsonan dasarnya sama tapi sandhangannya tertukar
            if pred_details.get("base_consonant") == target_details.get("base_consonant"):
                is_close = True
                verdict = "SANDHANGAN_MISMATCH"
                feedback = (f"Aksara dasar sudah benar ({target_details['base_consonant']}), "
                            f"namun sandhangan tertukar dengan {pred_details['category_name']}.")
            else:
                verdict = "INCORRECT"
                feedback = (f"Kurang tepat. Model mendeteksi aksara '{pred_details['latin']}' "
                            f"({pred_details['unicode_char']}), sedangkan target yang diminta adalah '{target_details['latin']}' ({target_details['unicode_char']}).")

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "status": "success",
            "verdict": verdict,
            "is_correct": is_exact,
            "is_close": is_close,
            "confidence": predicted_conf,
            "feedback": feedback,
            "predicted": pred_details,
            "target": target_details,
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
