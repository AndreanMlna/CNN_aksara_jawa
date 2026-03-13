# File: evaluate.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import modul yang sudah kita buat di folder core
from core import config, data_handler, model_builder


def evaluate_specific_model(model_path, version_name, val_loader, num_classes, label_to_idx, device):
    """
    Fungsi helper untuk mengevaluasi satu spesifik model dan mengembalikan daftar kesalahannya.
    """
    print(f"\n" + "=" * 50)
    print(f"📊 EVALUASI MODEL: {version_name}")
    print(f"Lokasi: {model_path}")
    print("=" * 50)

    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan di {model_path}!")
        return None

    # Inisialisasi arsitektur dan muat bobot
    model = model_builder.build_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Melakukan Prediksi (Inference)...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    # Pastikan urutan label sinkron
    sorted_labels_by_idx = sorted(label_to_idx.items(), key=lambda item: item[1])
    class_labels = [item[0] for item in sorted_labels_by_idx]

    # Hitung Akurasi
    final_acc = accuracy_score(y_true, y_pred)
    print(f"✅ Akurasi Keseluruhan ({version_name}): {final_acc * 100:.2f}%")

    print(f"\nClassification Report ({version_name}):")
    print(classification_report(y_true, y_pred, target_names=class_labels, labels=range(num_classes)))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)

    plt.title(f'Confusion Matrix Aksara Jawa - {version_name}\n(Accuracy: {final_acc * 100:.2f}%)', fontsize=18)
    plt.ylabel('Label Asli', fontsize=12)
    plt.xlabel('Prediksi Model', fontsize=12)

    if not os.path.exists(config.PLOT_DIR):
        os.makedirs(config.PLOT_DIR)

    # Simpan dengan nama yang dinamis agar tidak saling timpa
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

    return errors


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    print("1. Memuat Data Validasi (Hanya dilakukan sekali agar cepat)...")
    _, val_df, num_classes, label_to_idx = data_handler.load_and_split_data()
    _, val_loader = data_handler.get_dataloaders(None, val_df, label_to_idx)

    # Tentukan Path Kedua Model
    model_v1_path = config.MODEL_SAVE_PATH  # Model Awal (aksara_efficientnet_final.pth)
    model_v2_path = os.path.join(config.MODEL_DIR, 'aksara_efficientnet_v2_finetuned.pth')  # Model Baru

    # 2. Jalankan Evaluasi untuk V1
    errors_v1 = evaluate_specific_model(model_v1_path, "V1_Awal", val_loader, num_classes, label_to_idx, device)

    # 3. Jalankan Evaluasi untuk V2
    errors_v2 = evaluate_specific_model(model_v2_path, "V2_FineTuned", val_loader, num_classes, label_to_idx, device)

    # 4. Tampilkan Tabel Perbandingan Khusus Kesalahan
    if errors_v1 is not None and errors_v2 is not None:
        print("\n" + "=" * 65)
        print("🏆 PERBANDINGAN KESALAHAN (V1 vs V2)")
        print("=" * 65)
        print(f"{'Aksara':<25} | {'Salah di V1':<12} | {'Salah di V2':<12} | {'Status'}")
        print("-" * 65)

        # Gabungkan semua nama huruf yang punya error di V1 maupun V2
        all_error_keys = set(errors_v1.keys()).union(set(errors_v2.keys()))

        for aksara in sorted(all_error_keys):
            err1 = errors_v1.get(aksara, 0)
            err2 = errors_v2.get(aksara, 0)

            if err2 < err1:
                status = "✅ Membaik"
            elif err2 > err1:
                status = "⚠️ Memburuk"
            else:
                status = "➖ Tetap"

            print(f"{aksara:<25} | {err1:<12} | {err2:<12} | {status}")
        print("=" * 65)


if __name__ == '__main__':
    evaluate_model()