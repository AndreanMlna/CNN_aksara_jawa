import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import modul inti
from core import config, data_handler, model_builder


def get_model_errors(model_path, val_loader, num_classes, label_to_idx, device):
    """Fungsi super cepat khusus untuk mengambil jumlah salah DAN total per aksara."""
    if not os.path.exists(model_path):
        return None, None

    model = model_builder.build_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Susun label sesuai index
    sorted_labels = sorted(label_to_idx.items(), key=lambda item: item[1])
    class_labels = [item[0] for item in sorted_labels]

    # Hitung error dan total per kelas
    errors = {}
    totals = {}

    for i in range(num_classes):
        class_mask = (y_true == i)
        total_count = len(np.where(class_mask)[0])
        wrong_count = len(np.where(class_mask & (y_pred != i))[0])

        if total_count > 0:
            totals[class_labels[i]] = total_count
            if wrong_count > 0:
                errors[class_labels[i]] = wrong_count

    return errors, totals


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai analisis grafis Model V2 di: {device}")

    # 1. Load Data
    _, val_df, num_classes, label_to_idx = data_handler.load_and_split_data()
    _, val_loader = data_handler.get_dataloaders(None, val_df, label_to_idx)

    # Hanya ambil Model V2
    model_v2_path = getattr(config, 'MODEL_V2_SAVE_PATH',
                            os.path.join(config.MODEL_DIR, 'aksara_efficientnet_v2_finetuned.pth'))

    # 2. Ambil data error dan total
    print("⏳ Menghitung kesalahan pada Model V2 (Fine-Tuned)...")
    errors_v2, totals_v2 = get_model_errors(model_v2_path, val_loader, num_classes, label_to_idx, device)

    if errors_v2 is None:
        print(f"❌ Model V2 tidak ditemukan di {model_v2_path}")
        return

    if not errors_v2:
        print("🎉 LUAR BIASA! Model Anda mendapat akurasi 100%, tidak ada error untuk diplot.")
        return

    # 3. Proses Data untuk Visualisasi
    # Urutkan berdasarkan jumlah salah (Tertinggi ke Terendah)
    sorted_errors = sorted(errors_v2.items(), key=lambda item: item[1], reverse=True)

    # Ambil Top 25 huruf paling bermasalah (jika errornya lebih dari 25 kelas)
    top_errors = sorted_errors[:25]

    labels = [item[0] for item in top_errors]
    error_counts = [item[1] for item in top_errors]
    total_counts = [totals_v2[label] for label in labels]

    # 4. Membuat Visualisasi Overlapping Bar Chart
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 10))

    y_pos = np.arange(len(labels))

    # Bar untuk Total Dataset (Background - Abu-abu)
    ax.barh(y_pos, total_counts, color='lightgray', edgecolor='gray', height=0.6, label='Total Gambar (Validasi)')

    # Bar untuk Kesalahan (Foreground - Merah)
    rects_errors = ax.barh(y_pos, error_counts, color='salmon', edgecolor='black', height=0.6, label='Jumlah Salah')

    # Label & Judul
    ax.set_xlabel('Jumlah Gambar', fontsize=12, fontweight='bold')
    ax.set_title('Proporsi Kesalahan Prediksi vs Total Dataset Validasi\n(Model V2 - Top 25 Aksara)', fontsize=16,
                 fontweight='bold', pad=20)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()  # Yang salahnya paling banyak taruh di paling atas
    ax.legend(fontsize=12, loc='lower right')

    # Set batas X agar teks tidak terpotong
    max_total = max(total_counts)
    ax.set_xlim(0, max_total + (max_total * 0.25))

    # Tambahkan teks informasi di ujung setiap bar (Contoh: "12 / 124 (9.6%)")
    for i, rect in enumerate(rects_errors):
        err = int(rect.get_width())
        tot = int(total_counts[i])
        persentase = (err / tot) * 100

        ax.annotate(f'{err} / {tot}  ({persentase:.1f}%)',
                    xy=(tot, rect.get_y() + rect.get_height() / 2),
                    xytext=(10, 0),  # Offset ke kanan dari ujung bar total
                    textcoords="offset points",
                    ha='left', va='center', fontsize=11, fontweight='bold', color='black')

    # Simpan Gambar
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    plot_path = os.path.join(config.PLOT_DIR, 'error_ratio_analysis_v2.png')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n✅ Selesai! Grafik analisis proporsi error berhasil disimpan di:\n📁 {plot_path}")


if __name__ == '__main__':
    main()