# =========================
# INSTALL
# =========================
# !pip install -q transformers datasets peft accelerate sentencepiece

# =========================
# IMPORTS
# =========================
import os
import time
import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
)

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# =========================
# CONFIG
# =========================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAIN_FILE = "/kaggle/input/train-distilled/train_distilled.json"

OUTPUT_DIR = "/kaggle/working/student_lora"

MAX_LENGTH = 1024
BATCH_SIZE = 4
EPOCHS = 2
LR = 2e-4
WARMUP_RATIO = 0.05

# =========================
# LOAD DATA
# =========================

dataset = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]
print("Total samples:", len(dataset))

# Train/Validation Split (10%)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# =========================
# TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# TOKENIZATION
# =========================

def tokenize_function(example):
    messages = example["messages"]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True
    )

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = full_tokens["input_ids"]
    labels = input_ids.copy()

    prompt_len = len(prompt_tokens["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
    }

train_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=train_dataset.column_names,
    num_proc=2
)

val_dataset = val_dataset.map(
    tokenize_function,
    remove_columns=val_dataset.column_names,
    num_proc=2
)

# =========================
# DATA COLLATOR
# =========================

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding=True,
    return_tensors="pt",
)

# =========================
# LOAD MODEL
# =========================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# =========================
# LORA CONFIG (~25M params)
# =========================

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 10 STEP TIME ESTIMATION
# =========================

print("\nRunning 10-step time estimation...\n")

estimation_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    max_steps=10,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    report_to="none",
)

estimation_trainer = Trainer(
    model=model,
    args=estimation_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

start_time = time.time()
estimation_trainer.train()
end_time = time.time()

seconds_for_10 = end_time - start_time
seconds_per_step = seconds_for_10 / 10

steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
total_steps = steps_per_epoch * EPOCHS

estimated_total_seconds = total_steps * seconds_per_step
estimated_hours = estimated_total_seconds / 3600

print("Seconds per step:", round(seconds_per_step, 3))
print("Steps per epoch:", steps_per_epoch)
print("Total steps:", total_steps)
print("Estimated total training hours:", round(estimated_hours, 2))
print("Kaggle limit: 12 hours\n")

# =========================
# FULL TRAINING WITH VALIDATION
# =========================

print("Starting full training...\n")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,

    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,

    logging_steps=20,

    # 🔥 IMPORTANT FIX
    eval_strategy="steps",
    save_strategy="steps",

    eval_steps=500,
    save_steps=500,

    load_best_model_at_end=True,

    save_total_limit=2,
    fp16=True,
    dataloader_num_workers=2,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()

# =========================
# SAVE LORA ADAPTER
# =========================

model.save_pretrained(OUTPUT_DIR)

print("\n✅ Student training complete.")
print("Saved to:", OUTPUT_DIR)