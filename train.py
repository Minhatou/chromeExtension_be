"""
Phase III — LoRA/QLoRA fine-tuning of Llama-3.2-1B-Instruct on bilingual IT dataset.

Prerequisites:
  pip install -r requirements_train.txt

Run:
  python train.py

Output:
  LoRA adapter saved to ./models/lora-adapter/
  (Load with: python test_inference.py --lora ./models/lora-adapter)
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── Config ──────────────────────────────────────────────────────────────
BASE_MODEL   = "./models/llama-3.2-1b-instruct"
DATA_PATH    = "./data/train.jsonl"
OUTPUT_DIR   = "./models/lora-adapter"
LOGS_DIR     = "./logs"

LORA_R       = 16       # LoRA rank
LORA_ALPHA   = 32       # LoRA scaling
LORA_DROPOUT = 0.05

BATCH_SIZE          = 4
GRAD_ACCUMULATION   = 4   # effective batch = 16
LEARNING_RATE       = 2e-4
EPOCHS              = 3
MAX_SEQ_LEN         = 512
# ────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert IT translator. Translate the given text into Vietnamese, "
    "considering the provided context to accurately disambiguate terminology. "
    "Reply with only the Vietnamese translation, nothing else."
)


def format_prompt(sample: dict) -> str:
    """Convert a JSONL sample into an instruction-tuning prompt string."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context: {sample['context']}\n"
        f"Text: {sample['input']}\n"
        f"Translation: {sample['output']}"
    )


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    print("Applying LoRA adapter config...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset from: {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(lambda x: {"text": format_prompt(x)})

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGS_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        optim="paged_adamw_8bit",   # memory-efficient optimizer for QLoRA
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",           # disable wandb/mlflow
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving LoRA adapter to: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! Run: python test_inference.py --lora ./models/lora-adapter")


if __name__ == "__main__":
    main()
