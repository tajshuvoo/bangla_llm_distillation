# =========================
# INSTALL
# =========================
# !pip install -q transformers datasets bitsandbytes accelerate sentencepiece

# =========================
# IMPORTS
# =========================
import os
import time
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# =========================
# CONFIG
# =========================

MODEL_ID = "tajshuvo/Bangla-Mistral-7B-Instruct-v0.2"
TRAIN_FILE = "/kaggle/input/bangla-qa/train.json"

OUTPUT_PATH = "/kaggle/working/train_distilled_part.json"

MAX_LENGTH = 1024
MAX_NEW_TOKENS_CAP = 512
START_INDEX = 0            # 🔥 change for next runs
LIMIT_SAMPLES = 2000       # 🔥 change for next runs
BATCH_SIZE = 8             # safe for T4
SAVE_EVERY = 200

# =========================
# LOAD DATA
# =========================

dataset = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]

end_index = min(START_INDEX + LIMIT_SAMPLES, len(dataset))
dataset = dataset.select(range(START_INDEX, end_index))

print("Processing samples:", START_INDEX, "to", end_index-1)
print("Total selected:", len(dataset))

# =========================
# LOAD 4-BIT MODEL
# =========================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

# =========================
# CLEAN FUNCTION
# =========================

def clean_output(text):
    text = text.strip()

    # Remove instruction wrappers if repeated
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()

    if "[INST]" in text:
        text = text.split("[INST]")[-1].strip()

    # Remove system prompt echo
    if "You are a helpful Bangla assistant." in text:
        text = text.replace("You are a helpful Bangla assistant.", "").strip()

    return text.strip()

# =========================
# GENERATION LOOP
# =========================

distilled_data = []
start_time = time.time()

for batch_start in range(0, len(dataset), BATCH_SIZE):

    batch = dataset.select(
        range(batch_start, min(batch_start + BATCH_SIZE, len(dataset)))
    )

    prompt_texts = []
    prompt_messages_list = []

    for example in batch:
        messages = example["messages"]
        prompt_messages = messages[:-1]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_texts.append(prompt_text)
        prompt_messages_list.append(prompt_messages)

    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to("cuda")

    prompt_lengths = inputs["attention_mask"].sum(dim=1)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_CAP,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # =========================
    # PROCESS EACH SAMPLE
    # =========================

    for i in range(len(batch)):

        generated_tokens = outputs[i][prompt_lengths[i]:]
        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        generated_text = clean_output(generated_text)

        new_messages = prompt_messages_list[i] + [
            {"role": "assistant", "content": generated_text}
        ]

        distilled_data.append({"messages": new_messages})

        global_idx = START_INDEX + batch_start + i

        # Print only first 3 for sanity check
        if global_idx < START_INDEX + 3:
            print("\n==============================")
            print("SAMPLE", global_idx)
            print("ANSWER:\n", generated_text)
            print("==============================\n")

    # =========================
    # TIME ESTIMATION
    # =========================
    if batch_start == BATCH_SIZE * 3:
        elapsed = time.time() - start_time
        processed = batch_start + BATCH_SIZE
        per_sample = elapsed / processed
        total_estimated = per_sample * len(dataset)
        hours = total_estimated / 3600

        print("⏳ Avg seconds/sample:", round(per_sample, 2))
        print("⏳ Estimated total hours:", round(hours, 2))
        print("⚠ Kaggle limit: 12 hours\n")

    # =========================
    # PROGRESS LOG
    # =========================
    if (batch_start + BATCH_SIZE) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {min(batch_start+BATCH_SIZE, len(dataset))}/{len(dataset)} | Elapsed: {round(elapsed/60,2)} min")

    # =========================
    # SAVE INCREMENTALLY
    # =========================
    if (batch_start + BATCH_SIZE) % SAVE_EVERY == 0:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(distilled_data, f, ensure_ascii=False)
        print(f"💾 Saved progress at {batch_start+BATCH_SIZE} samples")

# =========================
# FINAL SAVE
# =========================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(distilled_data, f, ensure_ascii=False)

print("\n✅ Distillation complete.")
print("Saved to:", OUTPUT_PATH)