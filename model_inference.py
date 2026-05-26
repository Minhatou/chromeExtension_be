import os
import requests

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
    Online model loader placeholder.
    Reads HF_TOKEN from environment or .env file.
    """
    print("[INIT] Loading online model configurations...")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[INIT] HF_TOKEN not in environment, trying to read from .env file...")
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.strip().split("=", 1)[1]
                        print("[INIT] Found HF_TOKEN in .env file.")
        except Exception as e:
            print(f"[INIT WARNING] Failed to read .env file: {e}")
            pass
    else:
        print("[INIT] Found HF_TOKEN in environment variables.")
    
    if not hf_token:
        print("[INIT WARNING] HF_TOKEN is empty! API requests will be unauthorized and heavily rate-limited.")
        hf_token = ""
    else:
        masked = hf_token[:8] + "..." + hf_token[-4:] if len(hf_token) > 12 else "..."
        print(f"[INIT] Loaded HF_TOKEN successfully (Masked: {masked})")
    
    print("[INIT] Configured online model: Qwen/Qwen2.5-1.5B-Instruct hosted on Hugging Face API.")
    return "online_model", hf_token


def generate_translation(model, tokenizer, text, context="", target_lang="auto", glossary=None, glossary_mode="both"):
    """
    Translate text using Qwen/Qwen2.5-1.5B-Instruct hosted on Hugging Face Serverless Inference API.
    """
    import time
    
    # model here acts as a dummy/config, tokenizer contains our hf_token
    hf_token = tokenizer
    
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
    
    # Call HF Serverless Inference API via Chat Router endpoint
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        masked_token = hf_token[:8] + "..." + hf_token[-4:] if len(hf_token) > 12 else "..."
        print(f"  [API CALL] Using Authorization Token: {masked_token}")
    else:
        print("  [API CALL WARNING] Calling API without HF_TOKEN Authorization headers!")
        
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
        "messages": messages,
        "max_tokens": 1024 if target_lang in ("explain", "summarize") else 512,
        "temperature": 0.1,
    }
    
    detected_src = detect_language(text)
    print(f"  [ONLINE MODEL] detected_src={detected_src} → target={target_lang}")
    print(f"  [ONLINE MODEL] Request payload length: {len(str(payload))} characters.")
    print(f"  [ONLINE MODEL] Calling endpoint: {url}")
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        elapsed = time.time() - start_time
        print(f"  [ONLINE MODEL] HTTP status code: {response.status_code} (took {elapsed:.2f} seconds)")
        
        if response.status_code != 200:
            print(f"  [ONLINE MODEL ERROR] Raw API Response: {response.text}")
            raise Exception(f"Hugging Face API returned error status {response.status_code}: {response.text}")
        
        res_data = response.json()
        result = res_data["choices"][0]["message"]["content"].strip()
        print(f"  [ONLINE MODEL] Successful response. Token usage: {res_data.get('usage', 'N/A')}")
    except Exception as e:
        print(f"  [ONLINE MODEL EXCEPTION] {e}")
        raise e

    print(f"  [ONLINE MODEL] API output successfully parsed. Raw length: {len(result)} chars.")


    if target_lang == "explain" and not result.startswith("1. Định nghĩa:"):
        return "1. Định nghĩa: " + result
    elif target_lang == "summarize" and not result.startswith("-"):
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
    if target_lang not in ("explain", "summarize"):
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        return lines[0] if lines else result
        
    return result

