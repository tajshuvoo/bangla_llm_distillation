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

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# IMPORTANT: correct adapter path
LORA_PATH = "/kaggle/input/teacher-lora-adapter/models/teacher_lora"

OUTPUT_DIR = "/kaggle/working/teacher_merged"

HF_REPO_ID = "tajshuvo/Bangla-Mistral-7B-Instruct-v0.2"

# =========================
# LOGIN TO HF (KAGGLE SECRET REQUIRED)
# =========================

login(token="")

# =========================
# LOAD BASE MODEL (FULL PRECISION)
# =========================

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# =========================
# LOAD LORA ADAPTER
# =========================

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

# =========================
# MERGE LORA INTO BASE
# =========================

print("Merging LoRA weights...")

merged_model = model.merge_and_unload()

# =========================
# SAVE MERGED MODEL
# =========================

print("Saving merged model locally...")

merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
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

print("Upload complete.")