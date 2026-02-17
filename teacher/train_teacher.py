# # =========================
# # INSTALL
# # =========================
# !pip install -q transformers datasets peft bitsandbytes accelerate sentencepiece

# =========================
# IMPORTS
# =========================
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# =========================
# CONFIG
# =========================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

TRAIN_FILE = "/kaggle/input/bangla-qa/train.json"
VAL_FILE   = "/kaggle/input/bangla-qa/val.json"

OUTPUT_DIR = "/kaggle/working/models/teacher_lora"

MAX_LENGTH = 1024
BATCH_SIZE = 4         # 🚀 faster than 1
EPOCHS = 1              # must be 1 for 12h
LR = 2e-4

# =========================
# LOAD DATA
# =========================

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE
    }
)

# =========================
# TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# TOKENIZATION (FAST VERSION)
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

    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,      # 🚀 dynamic padding later
    )

    input_ids = full_tokens["input_ids"]
    labels = input_ids.copy()

    prompt_length = len(prompt_tokens["input_ids"])
    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
    }

dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset["train"].column_names,
    num_proc=2,      # 🚀 parallel preprocessing
)

# =========================
# DATA COLLATOR (DYNAMIC PADDING = FASTER)
# =========================

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding=True,
    return_tensors="pt",
)

# =========================
# MODEL (4-bit QLoRA)
# =========================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# ⚠️ gradient checkpointing OFF (slow on P100)
model.config.use_cache = False

# =========================
# LORA CONFIG
# =========================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# TRAINING SETTINGS (FAST)
# =========================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,   # 🚀 fastest
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    dataloader_num_workers=2,        # 🚀 speed
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

# Save only LoRA adapter (small)
model.save_pretrained(OUTPUT_DIR)

print("Training complete.")
