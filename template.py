import os

# Use current directory
base_path = "."

structure = {
    "configs": [
        "teacher.yaml",
        "student.yaml",
        "generation.yaml"
    ],
    "data": {
        "raw": ["bangla_instruction.json"],
        "processed_chat": ["train.json", "val.json"],
        "distilled_chat": ["train_distilled.json", "val_distilled.json"]
    },
    "models": [
        "teacher_lora",
        "teacher_merged",
        "student_lora"
    ],
    "teacher": [
        "train_teacher.py",
        "generate_distilled_data.py",
        "merge_lora.py"
    ],
    "student": [
        "train_student.py"
    ],
    "evaluation": [
        "inference.py",
        "compare_models.py",
        "latency_benchmark.py",
        "quality_eval.py"
    ],
    "utils": [
        "chat_format.py",
        "tokenizer_utils.py",
        "data_loader.py",
        "collator.py",
        "loss_mask.py",
        "seed.py"
    ],
    "scripts": [
        "run_teacher.sh",
        "run_distill.sh",
        "run_student.sh"
    ]
}

def create_structure(base_path, structure):
    for key, value in structure.items():
        current_path = os.path.join(base_path, key)

        if isinstance(value, dict):
            os.makedirs(current_path, exist_ok=True)
            create_structure(current_path, value)

        elif isinstance(value, list):
            os.makedirs(current_path, exist_ok=True)

            for item in value:
                item_path = os.path.join(current_path, item)

                if "." in item:  # file
                    with open(item_path, "w") as f:
                        pass
                else:  # folder
                    os.makedirs(item_path, exist_ok=True)

# Create folders/files in CURRENT directory
create_structure(base_path, structure)

# Create root-level files
open("requirements.txt", "a").close()
open("README.md", "a").close()

print("Project structure created in current directory!")
