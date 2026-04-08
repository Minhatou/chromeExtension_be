import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_name="meta-llama/Llama-3.2-1B-Instruct", lora_weights_path=None):
    """
    Load the base model and LoRA weights with quantization for RTX 3070Ti (8GB VRAM).
    """
    print(f"Loading base model: {base_model_name}")
    
    # Use 4-bit quantization as typical for 8GB VRAM GPUs (using bitsandbytes)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Quantization configurations require accelerate and bitsandbytes
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if lora_weights_path:
        print(f"Loading LoRA weights from: {lora_weights_path}")
        model = PeftModel.from_pretrained(model, lora_weights_path)
        
    return model, tokenizer

def generate_translation(model, tokenizer, text, context=""):
    """
    Generate translation using the model, considering the context.
    """
    # System prompt for translation
    system_prompt = "You are an expert IT translator. Translate the given text into Vietnamese, considering the provided context to accurately disambiguate terminology."
    
    # Construct prompt with context
    prompt = f"{system_prompt}\n\nContext: {context}\nText: {text}\nTranslation:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3, # Low temperature for more deterministic translation
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the translation part by removing the prompt
    translation = output_text.replace(prompt, "").strip()
    
    return translation
