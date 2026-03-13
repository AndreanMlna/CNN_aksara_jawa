# ꦄꦏ꧀ꦱꦫꦗꦮ | Aksara Jawa Image Classification (120 Classes)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)

Repositori ini berisi implementasi *End-to-End Deep Learning* untuk mengenali dan mengklasifikasikan **120 kelas Aksara Jawa**, yang mencakup aksara dasar, pasangan, dan sandhangan. Proyek ini dibangun menggunakan **PyTorch** dan telah dioptimasi untuk *deployment* ke platform *Mobile* (Android) via **TensorFlow Lite** serta *Web* via **FastAPI**.

## ✨ Fitur Utama (Key Features)

* **Arsitektur Canggih (State-of-the-Art):** Menggunakan **EfficientNet-B0** yang memberikan keseimbangan sempurna antara akurasi tinggi dan ukuran file yang ringan untuk inferensi di perangkat *mobile*.
* **Two-Stage Training Strategy:** Menerapkan teknik *Transfer Learning* sebagai *warm-up*, dilanjutkan dengan *Fine-Tuning* mendalam pada 80 layer terakhir untuk mengenali lekukan aksara/sandhangan yang sangat halus.
* **Penanganan Data Imbalance:** Menggunakan *Custom Class Weights* pada `CrossEntropyLoss` untuk memberikan penalti lebih besar pada huruf-huruf yang memiliki tingkat kemiripan tinggi atau sering salah ditebak (seperti suku, pepet, taling).
* **Konversi Lintas Framework yang Aman:** Pipeline ekspor otomatis dari PyTorch (`.pth`) ➡️ ONNX ➡️ TensorFlow SavedModel ➡️ TensorFlow Lite (`.tflite`).
* **Presisi Float32 TFLite:** Konversi TFLite dilakukan dengan mempertahankan presisi *Float32* untuk mencegah *Quantization Loss*, sehingga akurasi di Android tetap sama persis dengan akurasi model asli PyTorch (98.30%).
* **FastAPI Server Terintegrasi:** Dilengkapi dengan script REST API untuk *deployment* langsung ke *backend* aplikasi web/mobile.

## 📊 Hasil Evaluasi (Performance Metrics)

Model dievaluasi secara ketat menggunakan *Validation Dataset* terpisah. Berikut adalah hasil performa terakhir:

* **Akurasi Model PyTorch (`.pth`):** `98.30%`
* **Akurasi Model TFLite (`.tflite`):** `98.30%` *(Tidak ada penurunan akurasi berkat optimasi Float32)*

## 📂 Struktur Direktori

```text
📦 CNN-Aksara-Jawa
 ┣ 📂 core/                     # Modul inti (config, data_handler, model_builder, utils)
 ┣ 📂 dataset/                  # Folder untuk data training dan validasi
 ┣ 📂 output/                   # Folder hasil training (model, logs, plots)
 ┣ 📜 train.py                  # Script utama untuk proses Two-Stage Training
 ┣ 📜 evaluate_tflite_vs_pth.py # Script komparasi akurasi PyTorch vs TFLite
 ┣ 📜 export_tflite.py          # Script konversi model ke TensorFlow Lite
 ┣ 📜 test_tflite_gui.py        # GUI interaktif untuk tes inference model TFLite lokal
 ┣ 📜 api_server.py             # Script server FastAPI untuk integrasi frontend
 ┣ 📜 requirements.txt          # Daftar dependensi library Python
 ┗ 📜 README.md                 # Dokumentasi proyek
🚀 Cara Penggunaan (How to Use)
1. Prasyarat (Prerequisites)
Pastikan Anda menggunakan Python 3.9 atau 3.10. Lakukan instalasi seluruh library pendukung dengan menjalankan:

Bash
pip install -r requirements.txt
(Catatan: Proyek ini menggunakan tensorflow==2.10, onnx==1.13.0, dan onnx-tf untuk stabilitas konversi).

2. Melatih Model (Training)
Untuk melatih model menggunakan pendekatan Two-Stage Training (Warm-up & Fine-Tuning):

Bash
python train.py
3. Ekspor ke Android (TensorFlow Lite)
Ubah model .pth menjadi .tflite agar bisa ditanamkan secara native ke Android Studio:

Bash
python export_tflite.py
File hasil konversi akan otomatis tersimpan di output/models/aksara_efficientnet_v2.tflite.

4. Uji Coba Model Lokal (GUI Testing)
Untuk mengetes model TFLite yang sudah dibuat dengan memilih gambar secara langsung dari komputer:

Bash
python test_tflite_gui.py
5. Menjalankan REST API Server
Jika ingin menghubungkan model ke Website atau aplikasi klien lain via protokol HTTP:

Bash
uvicorn api_server:app --reload
Akses dokumentasi interaktif (Swagger UI) di browser Anda melalui: http://127.0.0.1:8000/docs.
