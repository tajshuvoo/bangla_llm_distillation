# =========================
# INSTALL
# =========================
# !pip install -q transformers datasets evaluate bitsandbytes accelerate sentencepiece

# =========================
# IMPORTS
# =========================
import torch
import time
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# =========================
# CONFIG
# =========================

VAL_FILE = "/kaggle/input/datasets/tajshuvo/bangla-qa/val.json"
NUM_SAMPLES = 400
MAX_NEW_TOKENS = 512

MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tajshuvo/Bangla-TinyLlama-1.1B-Distilled",
    "tajshuvo/Bangla-Mistral-7B-Instruct-v0.2"
]

# 4-bit config ONLY for 7B
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# =========================
# LOAD DATA
# =========================

dataset = load_dataset("json", data_files={"val": VAL_FILE})["val"]
dataset = dataset.select(range(NUM_SAMPLES))

print("Evaluating samples:", len(dataset))

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# =========================
# GENERATION FUNCTION (FIXED)
# =========================

def generate_answer(tokenizer, model, model_name, messages):

    # ---------- TinyLlama (simple prompt) ----------
    if "TinyLlama" in model_name and "Mistral" not in model_name:
        question = messages[1]["content"]
        prompt = f"প্রশ্ন: {question}\nউত্তর:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # ---------- Mistral (proper chat template) ----------
    else:
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],        # exclude GT answer
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt"
        ).to(model.device)

    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
        )

    # 🔥 decode only generated tokens
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return text.strip()


# =========================
# EVALUATION LOOP
# =========================

for model_name in MODELS:

    print("\n" + "=" * 80)
    print("Evaluating:", model_name)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Load model ----------
    if "7B" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    model.eval()

    preds = []
    refs = []

    start_time = time.time()

    for i, example in enumerate(dataset):

        messages = example["messages"]
        ground_truth = messages[-1]["content"].strip()

        answer = generate_answer(tokenizer, model, model_name, messages)

        preds.append(answer)
        refs.append(ground_truth)

        # Show first 2 FULL examples (no truncation)
        if i < 2:
            print("\n----------------------------")
            print("QUESTION:\n", messages[1]["content"])
            print("\nGROUND TRUTH:\n", ground_truth)
            print("\nPREDICTION:\n", answer)
            print("----------------------------")

        # Time estimation after 5 samples
        if i == 4:
            elapsed = time.time() - start_time
            per_sample = elapsed / 5
            total_estimated = per_sample * NUM_SAMPLES
            print("\nEstimated total evaluation time:",
                  round(total_estimated/60, 2), "minutes\n")

    # =========================
    # METRICS
    # =========================

    bleu_score = bleu.compute(
        predictions=preds,
        references=[[r] for r in refs]
    )["bleu"]

    rouge_score = rouge.compute(
        predictions=preds,
        references=refs
    )["rougeL"]

    print("\nRESULTS for", model_name)
    print("BLEU:", round(bleu_score, 4))
    print("ROUGE-L:", round(rouge_score, 4))

    # Clean GPU
    del model
    torch.cuda.empty_cache()