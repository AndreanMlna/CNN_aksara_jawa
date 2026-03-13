import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog

from core import config, model_builder


def load_class_mapping():
    # Memuat mapping yang disimpan saat training
    with open(config.CLASS_MAP_PATH, 'r') as f:
        class_indices = json.load(f)
    # Balik dictionary dari {nama_kelas: index} menjadi {index: nama_kelas}
    return {int(v): k for k, v in class_indices.items()}


# OPTIMASI: target_kelas dibuat = None agar kita bisa pakai mode "Tebak Bebas"
def cek_tulisan(img_path, target_kelas=None, confidence_threshold=0.70):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Class Mapping untuk mendapatkan jumlah kelas
    class_map = load_class_mapping()
    num_classes = len(class_map)

    # 2. Inisialisasi Model & Load Weights
    model = model_builder.build_model(num_classes)

    # Gunakan V2 jika ada, jika tidak fallback ke V1
    model_path = getattr(config, 'MODEL_V2_SAVE_PATH', config.MODEL_SAVE_PATH)
    if not os.path.exists(model_path):
        model_path = config.MODEL_SAVE_PATH

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Preprocessing Gambar
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 4. Prediksi
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    confidence, idx_tertinggi = torch.max(probabilities, dim=0)
    confidence = confidence.item()
    idx_tertinggi = idx_tertinggi.item()

    kelas_terprediksi = class_map[idx_tertinggi]

    # 5. Logika Output Fleksibel
    print("\n" + "=" * 50)
    if target_kelas:
        print(f"Target yang diminta : {target_kelas}")
    print(f"Hasil deteksi AI    : {kelas_terprediksi} (Confidence: {confidence:.2%})")
    print("=" * 50)

    # Jika user memasukkan target, berikan evaluasi BENAR/SALAH
    if target_kelas:
        if kelas_terprediksi == target_kelas and confidence >= confidence_threshold:
            print("✅ HASIL: BENAR! Tulisan Anda bagus.")
        elif kelas_terprediksi == target_kelas and confidence < confidence_threshold:
            print("⚠️ HASIL: KURANG TEPAT. Mirip, tapi coba tulis lebih jelas.")
        else:
            print(f"❌ HASIL: SALAH. Anda menulis {kelas_terprediksi}, bukan {target_kelas}.")


if __name__ == '__main__':
    # Membuka jendela dialog bawaan Windows/OS untuk memilih file
    root = tk.Tk()
    root.withdraw()  # Sembunyikan jendela utama tkinter, cukup tampilkan pop-up file saja

    print("Silakan pilih gambar aksara Jawa yang ingin dites pada jendela yang muncul...")

    # Buka pop-up File Explorer
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar Aksara",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print("❌ Anda membatalkan pemilihan gambar.")
    else:
        print(f"📂 File dipilih: {file_path}")

        # Opsi: Minta user mengetik target, atau kosongkan untuk tebak otomatis
        target_input = input(
            "Masukkan nama target huruf (Contoh: aksara-dasar_ba) ATAU tekan Enter untuk tebak bebas: ").strip()

        if target_input == "":
            target_input = None  # Mode tebak bebas

        cek_tulisan(file_path, target_input)