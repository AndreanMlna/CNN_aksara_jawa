# File: test_tflite_gui.py
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog
import json
from core import config


def test_inference(image_path):
    # 1. Load Model TFLite
    model_path = os.path.join(config.MODEL_DIR, "aksara_efficientnet_v2.tflite")
    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan di: {model_path}")
        return

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Preprocessing Gambar (WAJIB SAMA DENGAN TRAINING)
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))

    # Konversi ke Float dan Skala 0-1
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    # NORMALISASI IMAGENET (Standard PyTorch/EfficientNet)
    # Ini yang sering membuat prediksi salah/rendah jika dilewatkan
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Ubah format ke (Batch, Channel, Height, Width) sesuai PyTorch
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array.astype(np.float32), axis=0)

    # 3. Jalankan Prediksi
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # 4. Ambil Hasil & Hitung Softmax
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Gunakan Softmax manual untuk akurasi persentase yang lebih baik
    exp_scores = np.exp(output_data[0] - np.max(output_data[0]))
    probabilities = exp_scores / exp_scores.sum()

    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]

    # 5. Load Label Langsung dari JSON agar Urutan 100% Akurat
    with open(config.CLASS_MAP_PATH, 'r') as f:
        class_indices = json.load(f)

    # Balik dictionary: {index: label_name}
    labels = {v: k for k, v in class_indices.items()}
    label_name = labels.get(predicted_class, "Unknown")

    print("\n" + "=" * 45)
    print(f"📁 File    : {os.path.basename(image_path)}")
    print(f"✅ Prediksi: {label_name.upper()}")
    print(f"🚀 Confidence: {confidence * 100:.2f}%")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("🖼️ Membuka jendela pilih gambar...")
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar Aksara Jawa",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if file_path:
        test_inference(file_path)
    else:
        print("❌ Tidak ada file yang dipilih.")