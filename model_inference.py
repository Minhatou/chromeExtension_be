import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_model(base_model_name="meta-llama/Llama-3.2-1B-Instruct", lora_weights_path=None):
    """
    Load the base model and LoRA weights with quantization for RTX 3070Ti (8GB VRAM).
    """
    print(f"Loading base model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Use BitsAndBytesConfig for 4-bit quantization (required in newer transformers)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if lora_weights_path:
        print(f"Loading LoRA weights from: {lora_weights_path}")
        model = PeftModel.from_pretrained(model, lora_weights_path)

    model.eval()
    return model, tokenizer

def generate_translation(model, tokenizer, text, context=""):
    """
    Generate translation using the model, considering the context.
    """
    system_prompt = (
        "You are a Vietnamese translator specializing in IT and software terminology. "
        "Your task: translate the given text into Vietnamese. "
        "Rules you MUST follow:\n"
        "- Output ONLY the Vietnamese translation. Nothing else.\n"
        "- Do NOT add notes, explanations, disclaimers, or any English text.\n"
        "- Do NOT repeat the original text.\n"
        "- Do NOT add phrases like 'Note:', 'Original Text:', 'Translation:', or any headers.\n"
        "- If a term has no Vietnamese equivalent, keep it in English.\n"
        "- Use IT-specific Vietnamese terminology when the context is technical."
    )

    prompt = (
        f"{system_prompt}\n\n"
        f"Context: {context}\n"
        f"Text to translate: {text}\n"
        f"Vietnamese translation:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,        # Low temperature = more deterministic
            do_sample=True,
            repetition_penalty=1.3, # Penalize repetition to cut filler text
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the part after the last "Vietnamese translation:" marker
    marker = "Vietnamese translation:"
    idx = output_text.rfind(marker)
    result = output_text[idx + len(marker):].strip() if idx != -1 else output_text.strip()

    # --- Post-processing: strip English filler the model appends ---
    # These patterns indicate end of actual translation content
    FILLER_MARKERS = [
        "Note:",
        "note:",
        "**",
        "Here is",
        "here is",
        "Original Text:",
        "original text:",
        "Translation:",
        "The translation",
        "Please let me know",
        "I have translated",
        "Let me know",
        "Additional",
        "Explanation:",
    ]

    # Find the earliest filler marker in the result and cut before it
    cut_at = len(result)
    for fm in FILLER_MARKERS:
        pos = result.find(fm)
        if pos != -1 and pos < cut_at:
            cut_at = pos

    result = result[:cut_at].strip()

    # If multiple lines, take only the first non-empty one as a safety fallback
    lines = [l.strip() for l in result.split("\n") if l.strip()]
    return lines[0] if lines else result

