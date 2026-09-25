import os
import json
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from core import config

def main():
    model_path = os.path.join(config.MODEL_DIR, 'aksara_efficientnet_v2.tflite')
    if not os.path.exists(model_path):
        print(f"Error: Model tidak ditemukan di {model_path}")
        return

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(config.CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    idx_to_label = {int(v): k for k, v in class_indices.items()}

    gu_dir = os.path.join('dataset', 'Javanese Script (Aksara Jawa) Dataset', 'suku', 'gu')
    image_files = glob.glob(os.path.join(gu_dir, '*.*'))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    correct = 0
    errors = []
    confidences = []
    target_class = 'suku_gu'

    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_array = (img_array - mean) / std
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            exp_scores = np.exp(output_data - np.max(output_data))
            probs = exp_scores / exp_scores.sum()

            pred_idx = int(np.argmax(probs))
            pred_label = idx_to_label.get(pred_idx, 'Unknown')
            conf = float(probs[pred_idx])

            target_idx = class_indices.get(target_class)
            target_conf = float(probs[target_idx]) if target_idx is not None else 0.0

            if pred_label == target_class:
                correct += 1
                confidences.append(conf)
            else:
                errors.append({
                    'file': os.path.basename(img_path),
                    'pred': pred_label,
                    'conf': round(conf * 100, 2),
                    'target_conf': round(target_conf * 100, 2)
                })
        except Exception as e:
            print(f"Error pada {img_path}: {e}")

    total = len(image_files)
    acc = (correct / total) * 100 if total > 0 else 0.0
    avg_conf = np.mean(confidences) * 100 if confidences else 0.0

    print("=" * 60)
    print("HASIL EVALUASI TFLITE PADA SEMUA CITRA KELAS 'suku_gu'")
    print("=" * 60)
    print(f"Total Citra Suku Gu dalam Dataset : {total}")
    print(f"Prediksi Benar (suku_gu)          : {correct} ({acc:.2f}%)")
    print(f"Total Kesalahan (Error)           : {len(errors)} ({(len(errors)/total)*100:.2f}%)")
    print(f"Rata-rata Keyakinan Prediksi Benar: {avg_conf:.2f}%")
    print("-" * 60)
    print("Distribusi Kelas Kesalahan:")
    err_dist = {}
    for e in errors:
        err_dist[e['pred']] = err_dist.get(e['pred'], 0) + 1
    for k, v in sorted(err_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {k}: {v} citra ({(v/total)*100:.2f}%)")

    print("-" * 60)
    print(f"Daftar Lengkap Kesalahan ({len(errors)} berkas):")
    for idx, e in enumerate(errors, 1):
        print(f"  [{idx}] {e['file']} -> Pred: {e['pred']} ({e['conf']}%), Target Gu: {e['target_conf']}%")
    print("=" * 60)

if __name__ == '__main__':
    main()
