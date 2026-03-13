import matplotlib.pyplot as plt
import os
import numpy as np
from . import config


# OPTIMASI: Tambahkan parameter 'filename' dengan nilai default
def plot_and_save_history(history1, history2, filename='training_history_refined.png'):
    """
    Visualisasi hasil training dengan penekanan pada titik optimal.
    """
    # Menggabungkan data dari Stage 1 dan Stage 2
    acc = history1['train_acc'] + history2['train_acc']
    val_acc = history1['val_acc'] + history2['val_acc']
    loss = history1['train_loss'] + history2['train_loss']
    val_loss = history1['val_loss'] + history2['val_loss']

    # Titik pemisah antara Stage 1 dan Stage 2
    split_point = len(history1['train_acc']) - 1
    epochs_range = range(len(acc))

    plt.style.use('ggplot')  # Menggunakan style yang lebih modern dan bersih
    plt.figure(figsize=(16, 6))

    # --- 1. Plot Akurasi ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='royalblue', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='darkorange', linewidth=2)

    # Tandai Akurasi Tertinggi
    best_acc_idx = np.argmax(val_acc)
    best_acc = val_acc[best_acc_idx]
    plt.scatter(best_acc_idx, best_acc, color='red', s=50, zorder=5)
    plt.annotate(f'Max: {best_acc:.2%}', (best_acc_idx, best_acc),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.axvline(x=split_point, color='gray', linestyle='--', alpha=0.7, label='Fine Tuning Start')
    plt.title('Training & Validation Accuracy', fontsize=14)
    plt.xlabel('Total Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # --- 2. Plot Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='royalblue', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='darkorange', linewidth=2)

    # Tandai Loss Terendah (Target 0.20)
    min_loss_idx = np.argmin(val_loss)
    min_loss = val_loss[min_loss_idx]
    plt.scatter(min_loss_idx, min_loss, color='red', s=50, zorder=5)
    plt.annotate(f'Min: {min_loss:.4f}', (min_loss_idx, min_loss),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.axvline(x=split_point, color='gray', linestyle='--', alpha=0.7, label='Fine Tuning Start')
    plt.title('Training & Validation Loss', fontsize=14)
    plt.xlabel('Total Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Simpan hasil plot
    if not os.path.exists(config.PLOT_DIR):
        os.makedirs(config.PLOT_DIR)

    # OPTIMASI: Gunakan variabel 'filename' yang dikirim dari train.py
    plot_path = os.path.join(config.PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)  # Resolusi tinggi untuk dokumen/laporan
    plt.close()  # Menutup plot untuk menghemat memori
    print(f"Grafik optimal disimpan di: {plot_path}")