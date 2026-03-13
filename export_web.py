# File: export_web.py

import torch
import os
from core import config, model_builder
import json


def export_to_onnx():
    print("Mengonversi model untuk Website (ONNX)...")

    with open(config.CLASS_MAP_PATH, 'r') as f:
        num_classes = len(json.load(f))

    # 1. Load Model V2
    model = model_builder.build_model(num_classes)
    model_path = getattr(config, 'MODEL_V2_SAVE_PATH', config.MODEL_SAVE_PATH)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 2. Dummy Input
    dummy_input = torch.randn(1, 3, 224, 224)

    export_path = os.path.join(config.MODEL_DIR, "aksara_efficientnet_v2_web.onnx")

    # 3. Export ke ONNX
    torch.onnx.export(
        model,
        (dummy_input,),  # <--- PERUBAHAN DI SINI: Tambahkan kurung dan koma
        export_path,
        export_params=True,
        opset_version=11,  # Versi standar yang stabil
        do_constant_folding=True,  # Optimasi struktur
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"✅ Selesai! File untuk Website tersimpan di: {export_path}")


if __name__ == "__main__":
    export_to_onnx()