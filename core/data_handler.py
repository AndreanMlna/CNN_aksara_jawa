import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from . import config


# 1. Load data tetap sama (Logika sudah bagus)
def load_and_split_data():
    filepaths = []
    labels = []

    for root, dirs, files in os.walk(config.BASE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                fp = os.path.join(root, file)
                filepaths.append(fp)
                huruf = os.path.basename(os.path.dirname(fp))
                kategori = os.path.basename(os.path.dirname(os.path.dirname(fp)))
                labels.append(f"{kategori}_{huruf}")

    df = pd.DataFrame({'filepath': filepaths, 'label': labels})

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )

    unique_labels = sorted(df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    return train_df, val_df, len(unique_labels), label_to_idx


# 2. Custom Dataset (Tetap sama)
class AksaraDataset(Dataset):
    def __init__(self, dataframe, label_to_idx, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'filepath']
        label_str = self.dataframe.loc[idx, 'label']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_idx = self.label_to_idx[label_str]
        return image, label_idx


# 3. Fungsi generator OPTIMAL (Ditambahkan Augmentasi Target)
def get_dataloaders(train_df, val_df, label_to_idx):
    # Statistik Normalisasi ImageNet
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # --- Transformasi Training ---
    train_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),

        # 1. Augmentasi Bentuk Dasar (Sangat membantu untuk tulisan tangan)
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),

        # 2. TAMBAHAN: Random Perspective
        # Mensimulasikan distorsi jika kertas tidak rata atau difoto sedikit miring.
        # Sangat membantu model agar tidak kaku pada bentuk baku.
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),

        # 3. Augmentasi Warna & Pencahayaan
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),  # Ubah ke Tensor DULU sebelum RandomErasing dan Normalize

        # 4. TAMBAHAN: Random Erasing (Sangat Ampuh untuk mengatasi huruf mirip)
        # Akan memblokir 2-10% area gambar. Memaksa AI melihat seluhur huruf,
        # tidak hanya fokus pada satu titik sandhangan (seperti pepet atau suku).
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),

        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # --- Transformasi Validasi (TIDAK BOLEH ADA AUGMENTASI) ---
    val_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # --- Logika Pencegahan Error NoneType (Tetap dipertahankan) ---
    train_loader = None
    if train_df is not None:
        train_dataset = AksaraDataset(train_df, label_to_idx, transform=train_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

    val_loader = None
    if val_df is not None:
        val_dataset = AksaraDataset(val_df, label_to_idx, transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

    return train_loader, val_loader