# File: config.py

import os
import torch

# Path Dataset (Sesuaikan dengan lokasi folder dataset di komputermu)
BASE_DIR = r'./dataset/Javanese Script (Aksara Jawa) Dataset'

# Path Output
OUTPUT_DIR = './output'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
CLASS_MAP_PATH = os.path.join(MODEL_DIR, 'class_indices.json')

# --- OPTIMASI PATH MODEL (Versioning) ---
# Path Model Awal (V1)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'aksara_efficientnet_final.pth')

# Path Model Fine-Tuned (V2) - Ditambahkan agar tidak hardcode di file lain
MODEL_V2_SAVE_PATH = os.path.join(MODEL_DIR, 'aksara_efficientnet_v2_finetuned.pth')

# --- OPTIMASI HYPERPARAMETERS ---
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Tahap 1: Warm Up (Hanya Classifier)
EPOCHS_STAGE_1 = 15

# Tahap 2: Fine-Tuning (Membuka layer backbone)
EPOCHS_STAGE_2 = 30

# --- OPTIMASI LEARNING RATE (Disesuaikan dengan hasil terbaik V2) ---
LEARNING_RATE_S1 = 1e-4  # Diturunkan dari 1e-3 agar tidak merusak kepintaran model V1 saat fine-tuning
LEARNING_RATE_S2 = 1e-5  # Sangat kecil agar penyesuaian pada sandhangan (suku/pepet) tidak overfit
WEIGHT_DECAY = 1e-2      # Untuk AdamW agar tidak overfit
LABEL_SMOOTHING = 0.1    # Mencegah overconfidence & menjaga kestabilan Loss

# Membuat folder output jika belum ada (Hanya dijalankan oleh proses utama)
if __name__ == "__main__" or os.environ.get("LOCAL_RANK", "0") == "0":
    for d in [MODEL_DIR, PLOT_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"Setup direktori selesai. Model akan disimpan di folder: {MODEL_DIR}")