import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from core import config, data_handler, model_builder, utils


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), correct / total


# PERUBAHAN: Menambahkan parameter save_path agar kita bisa menyimpan ke versi V2
def run_stage(model, train_loader, val_loader, epochs, optimizer, criterion, device, stage_name, save_path):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8 if "Fine" in stage_name else 4

    # PERBAIKAN WARNING: verbose=True dihapus agar log lebih bersih
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)

    print(f"\n--- Memulai {stage_name} ---")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']  # Ambil LR saat ini untuk ditampilkan

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} [LR: {current_lr:.2e}] - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)  # Simpan ke save_path (V2)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping di epoch {epoch + 1}")
                break
    return history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan Perangkat: {device}")

    # LOGIKA BARU: Penamaan File Model Lama dan Model Baru (Versioning)
    old_model_path = config.MODEL_SAVE_PATH
    new_model_path = os.path.join(config.MODEL_DIR, 'aksara_efficientnet_v2_finetuned.pth')

    print("1. Memuat Dataset...")
    train_df, val_df, num_classes, label_to_idx = data_handler.load_and_split_data()
    train_loader, val_loader = data_handler.get_dataloaders(train_df, val_df, label_to_idx)

    with open(config.CLASS_MAP_PATH, 'w') as f:
        json.dump(label_to_idx, f)

    print("\n2. Membangun Model EfficientNetB0...")
    model = model_builder.build_model(num_classes).to(device)

    # ------------------------------------------------------------------------
    # LOGIKA BARU: Memuat Bobot Model V1 (Akurasi 95%) sebagai titik awal
    # ------------------------------------------------------------------------
    if os.path.exists(old_model_path):
        print(f"✅ Memuat pengetahuan dari model sebelumnya: {old_model_path}")
        model.load_state_dict(torch.load(old_model_path, map_location=device))
    else:
        print("⚠️ Peringatan: Model awal tidak ditemukan, akan training dari nol.")

    # ------------------------------------------------------------------------
    # LOGIKA BARU: Class Weights (Menghukum model lebih keras jika salah di huruf lemah)
    # ------------------------------------------------------------------------
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Daftar huruf yang banyak salah dari evaluasi kamu
    weak_classes = [
        'suku_du', 'suku_nu', 'suku_wu', 'pepet_yê', 'taling_nge',
        'suku_lu', 'suku_ngu', 'pepet_pê', 'taling_be', 'pepet_tê',
        'taling_we', 'suku_cu', 'taling-tarung_bo', 'taling-tarung_po',
        'taling-tarung_ngo'
    ]

    class_weights = torch.ones(num_classes)
    for idx in range(num_classes):
        label_name = idx_to_label[idx]
        if label_name in weak_classes:
            class_weights[idx] = 2.5  # Penalti 2.5x lipat untuk huruf yang paling parah
        elif 'suku' in label_name or 'pepet' in label_name or 'taling' in label_name:
            class_weights[idx] = 1.5  # Penalti 1.5x lipat untuk sandhangan secara umum

    class_weights = class_weights.to(device)

    # Masukkan bobot ke Criterion Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # --- TAHAP 1: Warm Up (Penyesuaian Loss Function Baru) ---
    # Karena kita sudah punya model pintar, LR Warm Up diturunkan ke 1e-4 agar tidak merusak yang sudah bagus
    optimizer_s1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    history1 = run_stage(
        model, train_loader, val_loader, config.EPOCHS_STAGE_1,
        optimizer_s1, criterion, device, "Tahap 1 (Warm Up V2)", new_model_path
    )

    # --- TAHAP 2: Advanced Fine Tuning ---
    print("\n3. Membuka Layer Lebih Dalam untuk Fine Tuning Spesifik...")
    # OPTIMASI: Buka lebih banyak layer (80) agar model bisa mengenali lekukan sandhangan lebih detil
    for param in list(model.features.parameters())[-80:]:
        param.requires_grad = True

    optimizer_s2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-2)

    history2 = run_stage(
        model, train_loader, val_loader, config.EPOCHS_STAGE_2,
        optimizer_s2, criterion, device, "Tahap 2 (Fine Tuning V2)", new_model_path
    )

    print("\n4. Menyimpan Grafik...")
    utils.plot_and_save_history(history1, history2)
    print(f"✅ Training Lanjutan Selesai! Model V2 disimpan di {new_model_path}")


if __name__ == '__main__':
    main()