# Bangla LLM Distillation
[![Hugging Face Teacher](https://img.shields.io/badge/🤗%20Teacher-Bangla%20Mistral%207B-yellow)](https://huggingface.co/tajshuvo/Bangla-Mistral-7B-Instruct-v0.2)
[![Hugging Face Student](https://img.shields.io/badge/🤗%20Student-Bangla%20TinyLlama%201.1B-blue)](https://huggingface.co/tajshuvo/Bangla-TinyLlama-1.1B-Distilled)
[![Kaggle](https://img.shields.io/badge/🟦%20Kaggle-Training%20Environment-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

Distilling a Fine-Tuned 7B Bangla Instruct Model into a 1.1B Student using LoRA

---

## Overview

This repository implements a complete teacher–student knowledge distillation pipeline for Bangla large language models.

A fine-tuned 7B Bangla instruction model is used as a teacher to generate distilled supervision data. A 1.1B TinyLlama model is then trained using LoRA to imitate the teacher’s behavior under constrained computational resources.

### Primary Objectives

- Compress a 7B Bangla LLM into a 1.1B model  
- Retain instruction-following capability  
- Preserve reasoning performance under model compression  
- Operate within single-GPU constraints (Tesla T4)

This repository stores Kaggle-executed training and evaluation code in an organized, reproducible structure. All experiments were conducted on Kaggle GPUs and later structured here for clarity and maintainability.

---

## Models

| Role | Model |
|------|--------|
| Teacher (Fine-Tuned) | tajshuvo/Bangla-Mistral-7B-Instruct-v0.2 |
| Student (Base) | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Student (Distilled) | tajshuvo/Bangla-TinyLlama-1.1B-Distilled |

> **Important Note**  
> The 7B Mistral model was instruction fine-tuned on Bangla data before being used as a teacher for distillation.

---

## Teacher Fine-Tuning Dataset

The teacher model was fine-tuned using:

- **Dataset:** md-nishat-008/Bangla-Instruct (Hugging Face)

### Dataset Configuration

- 6000 total samples  
- 5400 training samples  
- 600 validation samples  

Due to research and GPU memory constraints, a subset of the full dataset was used. The goal was to demonstrate a reproducible distillation workflow under practical limitations rather than maximize raw performance.

---

## Project Structure

```bash
bangla_llm_distillation
├── data
│   ├── distilled_chat
│   └── processed_chat
├── teacher
│   ├── train_teacher.py
│   ├── generate_distilled_data.py
│   └── merge_lora.py
├── student
│   ├── train_student.py
│   └── merge_lora.py
├── evaluation
│   └── compare_models.py
└── scripts
```

### Workflow Stages

- Teacher fine-tuning  
- Distilled dataset generation  
- Student LoRA training  
- Adapter merging  
- Multi-model evaluation  

---

# Pipeline Description

## 1️⃣ Fine-Tune Teacher (7B)

- **Base:** Mistral-7B-Instruct  
- **Dataset:** md-nishat-008/Bangla-Instruct  
- 6000 samples (600 validation)  
- LoRA fine-tuning  
- 4-bit quantization (QLoRA)  
- Adapter merged after training  

> The teacher is not the raw Mistral model. It is instruction fine-tuned for Bangla tasks prior to distillation.

---

## 2️⃣ Generate Distilled Dataset

The fine-tuned teacher generates responses for a Bangla QA dataset.

Each training example follows structured chat format:

```json
{
  "messages": [
    {"role": "system"},
    {"role": "user"},
    {"role": "assistant"}
  ]
}
```

Generated outputs are saved to:

```
data/distilled_chat/train_distilled.json
```

---

## 3️⃣ Train Student (1.1B)

The student model is trained to imitate teacher outputs.

### Configuration Highlights

- **Base:** TinyLlama-1.1B-Chat  
- LoRA with ~25M trainable parameters  
- r = 32  
- lora_alpha = 64  

### Target Modules

- q_proj  
- k_proj  
- v_proj  
- o_proj  
- gate_proj  
- up_proj  
- down_proj  

### Training Setup

- Max sequence length: 1024  
- FP16 training  
- Gradient checkpointing  
- 2 epochs  
- ~5400 distilled training samples  
- Tesla T4 GPU  

**Trainable Parameters:**  
25,231,360 (~2.24% of total parameters)

---

## 4️⃣ Merge LoRA

After training, LoRA adapters are merged into the base model:

```python
model.merge_and_unload()
```

This produces a standalone distilled model for inference.

---

## 5️⃣ Evaluation

Evaluation was performed on a Bangla QA validation set using:

- BLEU  
- ROUGE-L  

### Models Compared

- TinyLlama base (1.1B)  
- Distilled 1.1B  
- Fine-tuned 7B teacher  

### Results (400 Validation Samples)

| Model | BLEU | ROUGE-L |
|-------|------|----------|
| TinyLlama Base (1.1B) | 0.005 | 0.032 |
| Distilled 1.1B | 0.038 | 0.154 |
| Fine-Tuned 7B Teacher | 0.185 | 0.313 |

---

## Interpretation

Absolute BLEU scores are modest due to:

- Long-form reasoning responses  
- Mathematical derivations  
- Variability in phrasing  
- Open-ended answer structure  

However, relative improvement is meaningful:

- Distillation improves ROUGE-L by approximately **4.8×** over base TinyLlama  
- Student retains roughly **50% of teacher performance**  
- Model size reduced from 7B to 1.1B (~6× smaller)  

This demonstrates a functioning compression pipeline under realistic GPU and dataset constraints.

---

# Resource Constraints and Limitations

Student performance is influenced by:

- Limited distilled dataset size (~5400 examples)  
- Only 2 training epochs  
- Tesla T4 GPU memory constraints  
- 1024 token context limit  
- Parameter-efficient LoRA instead of full fine-tuning  

With:

- Larger distilled datasets  
- Additional epochs  
- Higher LoRA rank  
- Stronger student backbone  
- Multi-stage distillation  

The same structural pipeline can produce substantially stronger compressed Bangla models.

This repository prioritizes demonstrating an end-to-end, reproducible engineering workflow under research-level constraints.

---

# Hardware Environment

- Kaggle Tesla T4 GPU  
- 4-bit quantization for 7B teacher  
- FP16 training for student  
- Single-GPU setup  

All experiments were executed on Kaggle before being organized in this repository.

---

# What This Project Demonstrates

- Fine-tuning a 7B Bangla instruction model  
- Knowledge distillation from large to small LLM  
- LoRA and QLoRA training under memory constraints  
- Structured evaluation and benchmarking  
- Practical teacher–student compression trade-offs  

---

# License

MIT License
