"""
Phase II & IV — Test IT translation before and after fine-tuning.

Run BEFORE training (baseline):
  python test_inference.py

Run AFTER training (with LoRA adapter):
  python test_inference.py --lora ./models/lora-adapter
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "./models/llama-3.2-1b-instruct"

TEST_CASES = [
    {
        "label": "cloud (IT context)",
        "text": "cloud storage",
        "context": "AWS documentation about scalable cloud services and compute resources.",
        "expected": "lưu trữ đám mây / điện toán đám mây",
    },
    {
        "label": "cloud (weather context)",
        "text": "cloud",
        "context": "Today's weather forecast shows heavy clouds and rain expected in the afternoon.",
        "expected": "đám mây (thời tiết)",
    },
    {
        "label": "node (DOM context)",
        "text": "node",
        "context": "Traversing the DOM tree to find and remove a child node in JavaScript.",
        "expected": "nút DOM",
    },
    {
        "label": "node (network context)",
        "text": "node",
        "context": "A peer node in a distributed blockchain network handling transactions.",
        "expected": "nút mạng",
    },
    {
        "label": "thread (OS context)",
        "text": "thread",
        "context": "Multi-threaded Python application using concurrent.futures.ThreadPoolExecutor.",
        "expected": "luồng",
    },
    {
        "label": "stack (data structure)",
        "text": "stack",
        "context": "Implementing a LIFO stack data structure for an algorithm problem.",
        "expected": "ngăn xếp",
    },
    {
        "label": "stack (tech stack)",
        "text": "stack",
        "context": "Our web application stack consists of React, Node.js, and PostgreSQL.",
        "expected": "bộ công nghệ",
    },
]

SYSTEM_PROMPT = (
    "You are an expert IT translator. Translate the given text into Vietnamese, "
    "considering the provided context to accurately disambiguate terminology. "
    "Reply with only the Vietnamese translation, nothing else."
)


def build_prompt(text: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context: {context}\n"
        f"Text: {text}\n"
        f"Translation:"
    )


def load(base_model: str, lora_path: str | None):
    print(f"Loading base model from: {base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def translate(model, tokenizer, text: str, context: str) -> str:
    prompt = build_prompt(text, context)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Strip the prompt prefix to get just the translation
    marker = "Translation:"
    idx = decoded.rfind(marker)
    if idx != -1:
        return decoded[idx + len(marker):].strip()
    return decoded.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter")
    args = parser.parse_args()

    mode = "POST-TRAINING (LoRA)" if args.lora else "BASELINE (No Fine-Tuning)"
    print(f"\n{'='*60}")
    print(f"  IT Translator — Inference Test [{mode}]")
    print(f"{'='*60}\n")

    model, tokenizer = load(BASE_MODEL, args.lora)

    results = []
    for tc in TEST_CASES:
        print(f"[{tc['label']}]")
        print(f"  Input   : {tc['text']}")
        print(f"  Context : {tc['context'][:70]}...")
        result = translate(model, tokenizer, tc["text"], tc["context"])
        print(f"  Output  : {result}")
        print(f"  Expected: {tc['expected']}")
        print()
        results.append({"label": tc["label"], "output": result, "expected": tc["expected"]})

    print(f"{'='*60}")
    print(f"  Total cases: {len(results)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
