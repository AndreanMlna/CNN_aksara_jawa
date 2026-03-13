# File: export_tflite.py

import os
import torch
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import json
import shutil

# Import modul inti kita
from core import config, model_builder


def export_to_tflite():
    # Menyiapkan Path
    tflite_path = os.path.join(config.MODEL_DIR, "aksara_efficientnet_v2.tflite")
    temp_onnx_path = os.path.join(config.MODEL_DIR, "temp_static.onnx")
    tf_saved_model_dir = os.path.join(config.MODEL_DIR, "tf_saved_model")

    print(f"🚀 Memulai jalur konversi: PyTorch ➡️ ONNX ➡️ TensorFlow ➡️ TFLite")

    # 1. Load Model PyTorch V2
    print("\n1. Memuat Model PyTorch V2...")
    with open(config.CLASS_MAP_PATH, 'r') as f:
        num_classes = len(json.load(f))

    device = torch.device('cpu')
    model = model_builder.build_model(num_classes)
    model_path = getattr(config, 'MODEL_V2_SAVE_PATH', config.MODEL_SAVE_PATH)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Export ke ONNX Statis
    print("2. Mengekspor ke ONNX Statis sementara (Tanpa Dynamic Axes)...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy_input,),
        temp_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    # 3. Konversi ONNX ke TensorFlow SavedModel
    print("3. Mengonversi ONNX ke TensorFlow SavedModel (Butuh waktu beberapa saat)...")
    onnx_model = onnx.load(temp_onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_saved_model_dir)

    # 4. Konversi TensorFlow SavedModel ke TFLite
    print("4. Mengonversi TensorFlow SavedModel menjadi TFLite (Murni Float32 untuk Akurasi Maksimal)...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)

    # PERBAIKAN: Fitur Quantization (Kompresi) dimatikan agar akurasi tidak anjlok 12%.
    # File akan sedikit lebih besar (~20MB) namun mempertahankan kecerdasan asli PyTorch.
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # 5. Menyimpan File
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"\n✅ Selesai! File TFLite siap pakai untuk Android Studio tersimpan di:\n📁 {tflite_path}")

    # Membersihkan file sementara agar folder output kamu tetap rapi
    if os.path.exists(temp_onnx_path):
        os.remove(temp_onnx_path)
    if os.path.exists(tf_saved_model_dir):
        shutil.rmtree(tf_saved_model_dir)


if __name__ == "__main__":
    export_to_tflite()