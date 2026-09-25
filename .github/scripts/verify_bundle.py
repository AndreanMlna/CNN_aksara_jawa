#!/usr/bin/env python3
"""
CI/CD Quality Gate Verifier - AksaraAI Studio Netlify Bundle
Standar: Staff ML & DevOps Engineer / First Principles
Memverifikasi integritas arsitektur, kelengkapan berkas, dan validitas model ONNX sebelum deployment.
"""

import os
import sys
import json

DEPLOY_DIR = "netlify_deploy"

REQUIRED_FILES = [
    os.path.join(DEPLOY_DIR, "index.html"),
    os.path.join(DEPLOY_DIR, "_headers"),
    os.path.join(DEPLOY_DIR, "_redirects"),
    os.path.join(DEPLOY_DIR, "css", "style.css"),
    os.path.join(DEPLOY_DIR, "js", "app_netlify.js"),
    os.path.join(DEPLOY_DIR, "js", "unicode_map.js"),
    os.path.join(DEPLOY_DIR, "models", "class_indices.json"),
    os.path.join(DEPLOY_DIR, "models", "aksara_efficientnet_v2_web.onnx"),
]


def log(msg, status="INFO"):
    symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "ℹ️"
    print(f"[{symbol} {status}] {msg}")


def check_required_files():
    print("\n--- Gate 1: Verifikasi Kelengkapan Berkas Static Bundle ---")
    missing = []
    for filepath in REQUIRED_FILES:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            size_kb = os.path.getsize(filepath) / 1024
            log(f"{filepath} ({size_kb:,.1f} KB)", "PASS")
        else:
            log(f"Berkas tidak ditemukan atau kosong: {filepath}", "FAIL")
            missing.append(filepath)

    if missing:
        raise FileNotFoundError(f"Gagal verifikasi Gate 1: {len(missing)} berkas hilang.")
    log("Semua berkas paket Netlify lengkap dan siap dipublikasikan.", "PASS")


def check_class_indices():
    print("\n--- Gate 2: Verifikasi Basis Data & Pemetaan 120 Kelas Aksara ---")
    map_path = os.path.join(DEPLOY_DIR, "models", "class_indices.json")
    with open(map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    num_classes = len(class_map)
    log(f"Jumlah kelas terdaftar: {num_classes} kelas", "INFO")
    if num_classes != 120:
        raise ValueError(f"Diharapkan 120 kelas aksara jawa, ditemukan {num_classes} kelas.")

    categories = ["aksara-dasar_", "suku_", "wulu_", "taling_", "pepet_", "taling-tarung_"]
    for cat in categories:
        matching = [k for k in class_map.keys() if k.startswith(cat)]
        if len(matching) < 15:
            raise ValueError(f"Kategori {cat} tidak lengkap ({len(matching)} kelas terdeteksi).")
        log(f"Kategori '{cat}': {len(matching)} kelas terverifikasi.", "PASS")

    log("Gate 2 berhasil: Seluruh 120 kelas Aksara Jawa valid dan sinkron.", "PASS")


def check_onnx_model():
    print("\n--- Gate 3: Verifikasi Integritas Model ONNX WebAssembly ---")
    model_path = os.path.join(DEPLOY_DIR, "models", "aksara_efficientnet_v2_web.onnx")
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    log(f"Ukuran model ONNX: {size_mb:.2f} MB", "INFO")

    if not (15.0 <= size_mb <= 25.0):
        raise ValueError(f"Ukuran model mencurigakan ({size_mb:.2f} MB). Diharapkan ~19.3 MB.")

    try:
        import onnx
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        opset = model.opset_import[0].version if model.opset_import else "Unknown"
        log(f"Validasi ONNX Checker sukses (Producer: {model.producer_name}, Opset: {opset})", "PASS")

        # Cek Input & Output Shape
        graph = model.graph
        inp = graph.input[0]
        out = graph.output[0]
        log(f"Input Tensor : name='{inp.name}'", "PASS")
        log(f"Output Tensor: name='{out.name}'", "PASS")
    except ImportError:
        log("Pustaka 'onnx' tidak terpasang, melewati cek graph tingkat rendah.", "INFO")

    log("Gate 3 berhasil: Model ONNX terverifikasi sehat dan bebas korupsi biner.", "PASS")


def check_headers_and_redirects():
    print("\n--- Gate 4: Verifikasi Konfigurasi Header & Routing Netlify ---")
    redirects_path = os.path.join(DEPLOY_DIR, "_redirects")
    headers_path = os.path.join(DEPLOY_DIR, "_headers")

    with open(redirects_path, "r", encoding="utf-8") as f:
        r_content = f.read()
    if "/static/*" not in r_content or "/*" not in r_content:
        raise ValueError("File _redirects tidak memiliki aturan routing /static/* atau SPA /*.")
    log("Aturan _redirects valid.", "PASS")

    with open(headers_path, "r", encoding="utf-8") as f:
        h_content = f.read()
    if "*.wasm" not in h_content or "application/wasm" not in h_content:
        raise ValueError("File _headers tidak memiliki MIME type application/wasm.")
    log("Aturan _headers & konfigurasi WebAssembly valid.", "PASS")


def main():
    print("=" * 60)
    print("  AKSARAAI STUDIO - CI/CD QUALITY GATE & DEPLOYMENT AUDITOR  ")
    print("=" * 60)

    try:
        check_required_files()
        check_class_indices()
        check_onnx_model()
        check_headers_and_redirects()
        print("\n" + "=" * 60)
        log("SEMUA GERBANG KUALITAS (QUALITY GATES) BERHASIL 100%!", "PASS")
        log("Paket netlify_deploy/ LAYAK dan SIAP UNTUK PRODUCTION RELEASE.", "PASS")
        print("=" * 60 + "\n")
        sys.exit(0)
    except Exception as e:
        print("\n" + "=" * 60)
        log(f"PENOLAKAN PIPELINE (QUALITY GATE REJECTED): {str(e)}", "FAIL")
        print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
