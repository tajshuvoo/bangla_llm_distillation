# =========================
# INSTALL
# =========================
# !pip install -q transformers peft accelerate sentencepiece huggingface_hub

# =========================
# IMPORTS
# =========================
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# =========================
# CONFIG
# =========================

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LORA_PATH = "/kaggle/input/student-lora/student_lora"
OUTPUT_DIR = "/kaggle/working/student_merged"

HF_REPO_ID = "tajshuvo/Bangla-TinyLlama-1.1B-Distilled"

# =========================
# LOGIN TO HF
# =========================

# 👉 Use Kaggle secret: HF_TOKEN
login(token="")

# =========================
# LOAD BASE MODEL (FP16)
# =========================

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# =========================
# LOAD STUDENT LORA
# =========================

print("Loading Student LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

# =========================
# MERGE LORA
# =========================

print("Merging LoRA weights...")

merged_model = model.merge_and_unload()

# =========================
# SAVE LOCALLY
# =========================

print("Saving merged model locally...")

merged_model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=True,
    max_shard_size="2GB"
)

tokenizer.save_pretrained(OUTPUT_DIR)

print("Local merge complete.")

# =========================
# PUSH TO HUGGINGFACE
# =========================

print("Uploading to HuggingFace...")

merged_model.push_to_hub(
    HF_REPO_ID,
    max_shard_size="2GB"
)

tokenizer.push_to_hub(HF_REPO_ID)

print("🚀 Student merged model uploaded successfully.")