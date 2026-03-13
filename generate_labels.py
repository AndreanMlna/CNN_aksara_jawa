import json
import os
from core import config

def main():
    # Load mapping JSON yang sudah ada
    if not os.path.exists(config.CLASS_MAP_PATH):
        print(f"❌ File {config.CLASS_MAP_PATH} tidak ditemukan!")
        return

    with open(config.CLASS_MAP_PATH, 'r') as f:
        class_indices = json.load(f)

    # Urutkan berdasarkan index (value) dari yang terkecil (0) ke terbesar
    sorted_labels = sorted(class_indices.items(), key=lambda item: item[1])

    # Tentukan path untuk menyimpan labels.txt (di folder model)
    labels_path = os.path.join(config.MODEL_DIR, 'labels.txt')

    # Tulis ke dalam file txt
    with open(labels_path, 'w', encoding='utf-8') as f:
        for label, idx in sorted_labels:
            f.write(f"{label}\n")

    print(f"✅ Berhasil! File labels.txt sebanyak {len(sorted_labels)} kelas telah dibuat di:\n📁 {labels_path}")

if __name__ == '__main__':
    main()