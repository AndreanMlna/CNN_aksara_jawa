# File: export_android.py

import torch
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from core import config, model_builder
import json


def export_to_android():
    print("Mengonversi model untuk Android (TorchScript)...")

    with open(config.CLASS_MAP_PATH, 'r') as f:
        num_classes = len(json.load(f))

    # 1. Load Model V2
    model = model_builder.build_model(num_classes)
    model_path = getattr(config, 'MODEL_V2_SAVE_PATH', config.MODEL_SAVE_PATH)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 2. Buat Dummy Input (Tensor kosong berukuran sama dengan input aslimu)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. Tracing (Merekam jejak eksekusi model)
    traced_script_module = torch.jit.trace(model, dummy_input)

    # 4. Optimasi khusus Mobile (Mengurangi ukuran & mempercepat inferensi)
    optimized_mobile_model = optimize_for_mobile(traced_script_module)

    # 5. Simpan file .ptl
    export_path = os.path.join(config.MODEL_DIR, "aksara_efficientnet_v2_android.ptl")
    optimized_mobile_model._save_for_lite_interpreter(export_path)

    print(f"✅ Selesai! File untuk Android Studio tersimpan di: {export_path}")


if __name__ == "__main__":
    export_to_android()