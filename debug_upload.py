#!/usr/bin/env python3
"""Debug script to test HuggingFace upload step by step"""

from config import config
from src.huggingface_uploader import HuggingFaceUploader
from pathlib import Path

print("=== HUGGINGFACE UPLOAD DEBUG ===")
print(f"1. Config - enable_hf_upload: {config.enable_hf_upload}")
print(f"2. Config - hf_repo_name: {config.hf_repo_name}")
print(f"3. Config - huggingface_api_key set: {bool(config.huggingface_api_key)}")

print("\n4. Testing uploader initialization...")
uploader = HuggingFaceUploader()
print(f"   Uploader created: {uploader is not None}")

print("\n5. Testing authentication...")
auth_result = uploader.authenticate()
print(f"   Authentication result: {auth_result}")

print("\n6. Available datasets:")
datasets_dir = Path("datasets")
for file in datasets_dir.glob("*.json"):
    if not file.name.endswith('.metadata.json'):
        print(f"   - {file.name} ({file.stat().st_size} bytes)")

print("\n7. Testing dataset generator HF uploader...")
from src.dataset_generator import MedicalDatasetGenerator
gen = MedicalDatasetGenerator()
print(f"   Generator HF uploader: {gen.hf_uploader is not None}")
print(f"   Enable HF upload in config: {config.enable_hf_upload}")

print("\n=== DEBUG COMPLETE ===") 