import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Vietnamese diacritic characters for language detection
VI_CHARS = set(
    'àáảãạăắặẳẵâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
    'ÀÁẢÃẠĂẮẶẲẴÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ'
)

# Filler phrases small models tend to append after the translation
FILLER_MARKERS = [
    "Note:", "note:", "**", "Here is", "here is",
    "Original Text:", "original text:", "The translation",
    "Please let me know", "I have translated", "Let me know",
    "Additional", "Explanation:", "Context:", "I hope",
]


def detect_language(text: str) -> str:
    """Return 'vietnamese' if text contains VI diacritics, else 'english'."""
    return 'vietnamese' if any(c in VI_CHARS for c in text) else 'english'


def load_model(base_model_name="./models/llama-3.2-1b-instruct", lora_weights_path=None):
    """
    Load the base model and LoRA weights in full FP16 precision for RTX 3070Ti (8GB VRAM).
    This avoids dequantization overhead and speeds up generation significantly.
    """
    print(f"Loading base model in FP16: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if lora_weights_path:
        print(f"Loading LoRA weights from: {lora_weights_path}")
        model = PeftModel.from_pretrained(model, lora_weights_path)

    model.eval()
    return model, tokenizer


def generate_translation(model, tokenizer, text, context="", target_lang="auto", glossary=None, glossary_mode="both"):
    """
    Translate text with auto language detection (EN->VI or VI->EN), explain IT terms, or summarize text.
    target_lang: 'auto' | 'vietnamese' | 'english' | 'explain' | 'summarize'
    """
    # Rule A: Direct Selection Match (only if not in explain or summarize mode)
    if target_lang not in ("explain", "summarize") and glossary and glossary_mode in ("both", "direct"):
        normalized_glossary = {str(k).strip().lower(): str(v).strip() for k, v in glossary.items()}
        text_key = text.strip().lower()
        if text_key in normalized_glossary:
            meaning = normalized_glossary[text_key]
            print(f"  [GLOSSARY] Direct match found! {repr(text)} -> {repr(meaning)}")
            return meaning

    # Auto-detect source language and determine target
    if target_lang == "auto":
        src = detect_language(text)
        target_lang = "english" if src == "vietnamese" else "vietnamese"

    if target_lang not in ("vietnamese", "english", "explain", "summarize"):
        target_lang = "vietnamese"

    if target_lang == "vietnamese":
        lang_instruction = (
            "Translate the given text from English into Vietnamese. "
            "Use accurate IT-specific Vietnamese terminology when the context is technical."
        )
        system_prompt = (
            "Bạn là một biên dịch viên chuyên nghiệp về công nghệ thông tin. "
            "Nhiệm vụ của bạn là CHỈ dịch đoạn văn bản nằm trong khối [TEXT_TO_TRANSLATE] sang tiếng Việt. "
            "Khối [CONTEXT] được cung cấp CHỈ để giúp bạn hiểu rõ ngữ cảnh của từ ngữ hoặc các đại từ xưng hô, "
            "tuyệt đối KHÔNG được dịch các câu trong khối [CONTEXT] hay đưa bất kỳ nội dung nào từ [CONTEXT] vào kết quả đầu ra của bạn. "
            "Hãy CHỈ trả về bản dịch trực tiếp của văn bản trong khối [TEXT_TO_TRANSLATE], KHÔNG giải thích, không thêm tiêu đề, nhãn hay từ ngữ thừa nào khác."
        )
    elif target_lang == "english":
        lang_instruction = (
            "Translate the given text from Vietnamese into English. "
            "Use accurate IT-specific English terminology when the context is technical."
        )
        system_prompt = (
            "Bạn là một biên dịch viên chuyên nghiệp về công nghệ thông tin. "
            "Nhiệm vụ của bạn là CHỈ dịch đoạn văn bản nằm trong khối [TEXT_TO_TRANSLATE] sang tiếng Anh. "
            "Khối [CONTEXT] được cung cấp CHỈ để giúp bạn hiểu rõ ngữ cảnh của từ ngữ hoặc các đại từ xưng hô, "
            "tuyệt đối KHÔNG được dịch các câu trong khối [CONTEXT] hay đưa bất kỳ nội dung nào từ [CONTEXT] vào kết quả đầu ra của bạn. "
            "Hãy CHỈ trả về bản dịch trực tiếp của văn bản trong khối [TEXT_TO_TRANSLATE], KHÔNG giải thích, không thêm tiêu đề, nhãn hay từ ngữ thừa nào khác."
        )
    elif target_lang == "explain":
        lang_instruction = (
            "Giải thích thuật ngữ công nghệ thông tin trong khối [TERM] bằng tiếng Việt. "
            "Cung cấp định nghĩa rõ ràng, phân tích vai trò của nó trong lập trình và hệ thống."
        )
        system_prompt = (
            "Bạn là một giảng viên khoa học máy tính giỏi. "
            "Nhiệm vụ của bạn là giải thích thuật ngữ công nghệ nằm trong khối [TERM] bằng tiếng Việt. "
            "Tuyệt đối KHÔNG dịch nguyên văn đoạn văn bản, hãy giảng giải thuật ngữ. "
            "Cấu trúc phản hồi của bạn phải rõ ràng gồm:\n"
            "1. Định nghĩa ngắn gọn bằng tiếng Việt.\n"
            "2. Ý nghĩa/Vai trò của nó trong lập trình hoặc thiết kế hệ thống bằng tiếng Việt."
        )
    elif target_lang == "summarize":
        lang_instruction = (
            "Đọc hiểu đoạn văn bản công nghệ thông tin trong khối [DOCUMENT] và viết một bản tóm tắt cực kỳ ngắn gọn, súc tích bằng tiếng Việt. "
            "Tuyệt đối KHÔNG dịch nguyên văn, chỉ gạch đầu dòng tối đa 3-4 ý chính quan trọng nhất, mỗi ý không quá một câu ngắn."
        )
        system_prompt = (
            "Bạn là một trợ lý phân tích tài liệu kỹ thuật CNTT giỏi. "
            "Nhiệm vụ của bạn là đọc văn bản nằm trong khối [DOCUMENT] và viết một bản tóm tắt CỰC KỲ NGẮN GỌN các ý chính dưới dạng gạch đầu dòng bằng tiếng Việt. "
            "Tuyệt đối KHÔNG dịch nguyên văn và KHÔNG viết dài dòng. "
            "Chỉ liệt kê tối đa 3-4 gạch đầu dòng cốt lõi, mỗi gạch đầu dòng chỉ gồm 1 câu cực kỳ ngắn gọn, tập trung vào thông tin quan trọng nhất."
        )

    # Rule B: Prompt Context Enforced (only for non-explain and non-summarize modes)
    glossary_context = ""
    if target_lang not in ("explain", "summarize") and glossary and glossary_mode in ("both", "ai"):
        glossary_items = [f"- {k} -> {v}" for k, v in glossary.items() if str(k).strip().lower() in text.lower()]
        if glossary_items:
            glossary_context = "Glossary (Bắt buộc dùng các từ dịch này nếu xuất hiện trong văn bản):\n" + "\n".join(glossary_items)
            print(f"  [GLOSSARY] Enforced glossary prompt context:\n{glossary_context}")
            system_prompt += f"\n\n{glossary_context}"
    
    # Construct the messages for the chat template with strict tag wrapping
    if target_lang == "explain":
        user_content = (
            f"[TERM]\n{text}\n[/TERM]\n\n"
            f"Yêu cầu: {lang_instruction}\n"
            f"Giải thích chi tiết bằng tiếng Việt:\n"
            f"1. Định nghĩa: "
        )
    elif target_lang == "summarize":
        user_content = (
            f"[DOCUMENT]\n{text}\n[/DOCUMENT]\n\n"
            f"Yêu cầu: {lang_instruction}\n"
            f"Bản tóm tắt súc tích bằng tiếng Việt:\n"
            f"- "
        )
    else:
        user_content = (
            f"[CONTEXT]\n{context}\n[/CONTEXT]\n\n"
            f"[TEXT_TO_TRANSLATE]\n{text}\n[/TEXT_TO_TRANSLATE]\n\n"
            f"Instruction: {lang_instruction}\n"
            f"Result:"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Apply Qwen2 chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    token_count = inputs['input_ids'].shape[1]
    detected_src = detect_language(text)
    print(f"  [MODEL] detected_src={detected_src} → target={target_lang}")
    print(f"  [MODEL] input tokens: {token_count}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024 if target_lang in ("explain", "summarize") else 512,
            do_sample=False,  # Switch to greedy decoding for fast, consistent results
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode ONLY the newly generated tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print(f"  [MODEL] raw output: {repr(result)}")

    if target_lang == "explain":
        return "1. Định nghĩa: " + result
    elif target_lang == "summarize":
        return "- " + result

    # Cut off any English filler the model appends after the translation
    cut_at = len(result)
    for fm in FILLER_MARKERS:
        pos = result.find(fm)
        if pos != -1 and pos < cut_at:
            cut_at = pos

    result = result[:cut_at].strip()
    
    # Remove common prefixes generated by small models
    prefixes_to_remove = ["Tiêu đề bài viết:", "Sự dịch từ điển:", "Bản dịch:"]
    for p in prefixes_to_remove:
        if result.startswith(p):
            result = result[len(p):].strip()
            break

    # Return only the first non-empty line as a final safety fallback for translation
    lines = [l.strip() for l in result.split("\n") if l.strip()]
    return lines[0] if lines else result
