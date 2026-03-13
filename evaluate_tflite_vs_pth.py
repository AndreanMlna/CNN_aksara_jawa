# File: evaluate_tflite_vs_pth.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# Import modul yang sudah kita buat di folder core
from core import config, data_handler, model_builder


def evaluate_pytorch_model(model_path, version_name, val_loader, num_classes, label_to_idx, device):
    """Mengevaluasi model asli PyTorch (.pth)."""
    print(f"\n" + "=" * 50)
    print(f"📊 EVALUASI MODEL PYTORCH: {version_name}")
    print(f"Lokasi: {model_path}")
    print("=" * 50)

    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan di {model_path}!")
        return None, None

    model = model_builder.build_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Melakukan Prediksi (Inference)...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Inference {version_name}"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return process_results(all_labels, all_preds, num_classes, label_to_idx, version_name)


def evaluate_tflite_model(model_path, version_name, val_loader, num_classes, label_to_idx):
    """Mengevaluasi model TFLite (.tflite) satu per satu gambar."""
    print(f"\n" + "=" * 50)
    print(f"📱 EVALUASI MODEL TFLITE: {version_name}")
    print(f"Lokasi: {model_path}")
    print("=" * 50)

    if not os.path.exists(model_path):
        print(f"❌ Model TFLite tidak ditemukan di {model_path}!")
        return None, None

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Melakukan Prediksi (Inference TFLite)...")
    all_preds = []
    all_labels = []

    # Kita pakai val_loader yang sama agar datanya 100% adil (sudah dinormalisasi)
    for images, labels in tqdm(val_loader, desc=f"Inference {version_name}"):
        images_np = images.numpy()
        labels_np = labels.numpy()

        # TFLite di-export dengan batch size 1, jadi kita proses gambarnya satu-satu
        for i in range(images_np.shape[0]):
            input_data = np.expand_dims(images_np[i], axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            pred_class = np.argmax(output_data[0])
            all_preds.append(pred_class)
            all_labels.append(labels_np[i])

    return process_results(all_labels, all_preds, num_classes, label_to_idx, version_name)


def process_results(y_true, y_pred, num_classes, label_to_idx, version_name):
    """Fungsi pembantu untuk menghitung metrik, memplot CM, dan mencatat error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    sorted_labels_by_idx = sorted(label_to_idx.items(), key=lambda item: item[1])
    class_labels = [item[0] for item in sorted_labels_by_idx]

    final_acc = accuracy_score(y_true, y_pred)
    print(f"✅ Akurasi Keseluruhan ({version_name}): {final_acc * 100:.2f}%")

    print(f"\nClassification Report ({version_name}):")
    # Hanya print 10 kelas pertama agar terminal tidak terlalu penuh, atau matikan komentar di bawah jika butuh semua
    # print(classification_report(y_true, y_pred, target_names=class_labels, labels=range(num_classes)))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds' if 'TFLite' in version_name else 'Blues',
                xticklabels=class_labels, yticklabels=class_labels)

    plt.title(f'Confusion Matrix Aksara Jawa - {version_name}\n(Accuracy: {final_acc * 100:.2f}%)', fontsize=18)
    plt.ylabel('Label Asli', fontsize=12)
    plt.xlabel('Prediksi Model', fontsize=12)

    os.makedirs(config.PLOT_DIR, exist_ok=True)
    cm_plot_path = os.path.join(config.PLOT_DIR, f'confusion_matrix_{version_name}.png')
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()

    print(f"✅ Gambar Confusion Matrix disimpan di: {cm_plot_path}")

    # Catat Kesalahan
    errors = {}
    for i in range(num_classes):
        wrong_idx = np.where((y_true == i) & (y_pred != i))[0]
        if len(wrong_idx) > 0:
            errors[class_labels[i]] = len(wrong_idx)

    return errors, final_acc


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    print("1. Memuat Data Validasi...")
    _, val_df, num_classes, label_to_idx = data_handler.load_and_split_data()
    _, val_loader = data_handler.get_dataloaders(None, val_df, label_to_idx)

    # Path Model
    model_pth_path = os.path.join(config.MODEL_DIR, 'aksara_efficientnet_v2_finetuned.pth')
    model_tflite_path = os.path.join(config.MODEL_DIR, 'aksara_efficientnet_v2.tflite')

    # 2. Evaluasi V2 (PyTorch Asli)
    errors_pth, acc_pth = evaluate_pytorch_model(model_pth_path, "V2_PyTorch", val_loader, num_classes, label_to_idx,
                                                 device)

    # 3. Evaluasi TFLite (Android Version)
    errors_tflite, acc_tflite = evaluate_tflite_model(model_tflite_path, "V2_TFLite", val_loader, num_classes,
                                                      label_to_idx)

    # 4. Tampilkan Tabel Perbandingan
    if errors_pth is not None and errors_tflite is not None:
        print("\n" + "=" * 65)
        print("🏆 PERBANDINGAN PYTORCH vs TFLITE")
        print("=" * 65)
        print(f"Akurasi PyTorch Asli : {acc_pth * 100:.2f}%")
        print(f"Akurasi Versi TFLite : {acc_tflite * 100:.2f}%")

        diff = acc_pth - acc_tflite
        if diff > 0.01:
            print(f"⚠️ Terjadi sedikit penurunan akurasi sebesar {diff * 100:.2f}% akibat kompresi.")
        else:
            print(f"🎉 LUAR BIASA! Tidak ada penurunan akurasi yang signifikan saat di-convert!")

        print("-" * 65)
        print(f"{'Aksara':<25} | {'Salah PyTorch':<13} | {'Salah TFLite':<12} | {'Status'}")
        print("-" * 65)

        all_error_keys = set(errors_pth.keys()).union(set(errors_tflite.keys()))

        for aksara in sorted(all_error_keys):
            err1 = errors_pth.get(aksara, 0)
            err2 = errors_tflite.get(aksara, 0)

            if err2 < err1:
                status = "✅ Membaik (Unik)"
            elif err2 > err1:
                status = "⚠️ Korban Kompresi"
            else:
                status = "➖ Sama Persis"

            print(f"{aksara:<25} | {err1:<13} | {err2:<12} | {status}")
        print("=" * 65)


if __name__ == '__main__':
    evaluate_model()