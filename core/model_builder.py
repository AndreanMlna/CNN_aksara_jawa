import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes):
    # 1. Load Pre-trained EfficientNetB0
    # Menggunakan bobot ImageNet terbaru (V2 jika tersedia) untuk fitur ekstraksi terbaik
    weights = models.EfficientNet_B0_Weights.DEFAULT
    base_model = models.efficientnet_b0(weights=weights)

    # 2. Freeze base model secara total di awal
    for param in base_model.parameters():
        param.requires_grad = False

    # 3. Ambil jumlah input fitur (1280 untuk EfficientNet-B0)
    num_ftrs = base_model.classifier[1].in_features

    # 4. Membangun Custom Classifier yang Dioptimalkan
    # Menambahkan BatchNorm1d untuk menstabilkan distribusi aktivasi
    # Ini sangat membantu menurunkan Loss secara signifikan
    base_model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(num_ftrs, 512),  # Dinaikkan ke 512 untuk kapasitas memori pola yang lebih besar
        nn.BatchNorm1d(512),  # OPTIMASI: Menurunkan loss dan mempercepat konvergensi
        nn.ReLU(inplace=False),

        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(512, 256),  # Layer tambahan untuk ekstraksi fitur hirarki
        nn.BatchNorm1d(256),  # OPTIMASI: Menjaga stabilitas gradien
        nn.ReLU(inplace=False),

        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(256, num_classes)  # Output layer
        # CrossEntropyLoss akan menangani Softmax secara internal
    )

    return base_model