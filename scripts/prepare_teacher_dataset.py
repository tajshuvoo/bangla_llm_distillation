import os
import json
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# =========================
# CONFIG
# =========================

DATASET_NAME = "md-nishat-008/Bangla-Instruct"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

MAX_TOKENS = 1024        # Must match training max_seq_length
TARGET_SIZE = 6000       # Final dataset size
SEED = 42

OUTPUT_DIR = "data/processed_chat"

# =========================
# MAIN
# =========================

def main():

    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    filtered_samples = []

    print("Filtering by token length...")

    for sample in tqdm(dataset):

        # -------------------------
        # Safe null handling
        # -------------------------
        instruction = sample.get("instruction")
        response = sample.get("response")

        if not instruction or not response:
            continue

        instruction = instruction.strip()
        response = response.strip()

        if instruction == "" or response == "":
            continue

        # -------------------------
        # Build chat messages
        # -------------------------
        messages = [
            {"role": "system", "content": "You are a helpful Bangla assistant."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]

        # -------------------------
        # Apply chat template
        # -------------------------
        try:
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            continue

        # -------------------------
        # Tokenize without truncation
        # -------------------------
        try:
            tokens = tokenizer(
                chat_text,
                truncation=False,
                add_special_tokens=False
            )
        except Exception:
            continue

        token_length = len(tokens["input_ids"])

        if token_length <= MAX_TOKENS:
            filtered_samples.append({
                "messages": messages
            })

    print(f"\nTotal after filtering: {len(filtered_samples)}")

    if len(filtered_samples) < TARGET_SIZE:
        raise ValueError(
            f"Not enough filtered samples ({len(filtered_samples)}) "
            f"to reach target size {TARGET_SIZE}."
        )

    # -------------------------
    # Sampling
    # -------------------------
    print(f"Sampling exactly {TARGET_SIZE} rows...")
    random.seed(SEED)
    final_dataset = random.sample(filtered_samples, TARGET_SIZE)

    # Train/Val split
    split_index = int(0.9 * TARGET_SIZE)
    train_data = final_dataset[:split_index]
    val_data = final_dataset[split_index:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Saving files...")

    with open(f"{OUTPUT_DIR}/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(f"{OUTPUT_DIR}/val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print("\nDataset preparation complete!")
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")


if __name__ == "__main__":
    main()
