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


def load_model(base_model_name=r"C:\Users\Cko Ckeems Ngoo\LlamaFactory\qwen_7278", lora_weights_path=None):
    """
    Load model locally with 4-bit quantization for GPU VRAM efficiency.
    Falls back to online model configurations on failure.
    """
    print(f"[INIT] Attempting to load local model from: {base_model_name}...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        if lora_weights_path:
            print(f"[INIT] Loading LoRA weights from: {lora_weights_path}")
            model = PeftModel.from_pretrained(model, lora_weights_path)
            
        model.eval()
        print("[INIT] Local model loaded successfully!")
        return model, tokenizer
    except Exception as local_err:
        print(f"[INIT WARNING] Failed to load local model: {local_err}")
        print("[INIT] Falling back to online Hugging Face API configurations...")
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("HF_TOKEN="):
                            hf_token = line.strip().split("=", 1)[1]
            except Exception:
                pass
        return "online_model", hf_token


def generate_translation(model, tokenizer, text, context="", target_lang="auto", glossary=None, glossary_mode="both", model_id="qwen3"):
    """
    Translate text using Qwen models hosted on Hugging Face Serverless Inference API, Dedicated Endpoints, or locally.
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
            "Nếu có khối [GLOSSARY], hãy bắt buộc sử dụng các từ dịch được định nghĩa trong đó để dịch các từ tương ứng trong khối [TEXT_TO_TRANSLATE]. "
            "Hãy CHỈ trả về bản dịch trực tiếp của văn bản trong khối [TEXT_TO_TRANSLATE], KHÔNG giải thích, không dịch khối [GLOSSARY] hay khối [CONTEXT], không thêm tiêu đề, nhãn hay từ ngữ thừa nào khác."
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
            "Nếu có khối [GLOSSARY], hãy bắt buộc sử dụng các từ dịch được định nghĩa trong đó để dịch các từ tương ứng trong khối [TEXT_TO_TRANSLATE]. "
            "Hãy CHỈ trả về bản dịch trực tiếp của văn bản trong khối [TEXT_TO_TRANSLATE], KHÔNG giải thích, không dịch khối [GLOSSARY] hay khối [CONTEXT], không thêm tiêu đề, nhãn hay từ ngữ thừa nào khác."
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
        user_content = ""
        if glossary_context:
            user_content += f"[GLOSSARY]\n{glossary_context}\n[/GLOSSARY]\n\n"
        user_content += (
            f"[CONTEXT]\n{context}\n[/CONTEXT]\n\n"
            f"[TEXT_TO_TRANSLATE]\n{text}\n[/TEXT_TO_TRANSLATE]\n\n"
            f"Instruction: {lang_instruction}\n"
            f"Result:"
        )
    # Format as a Qwen2 chat template string manually
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    if model == "online_model":
        # Call HF Serverless Inference API via Chat completions router (DNS resolvable in this environment)
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
            
        if model_id == "qwen3":
            url = "https://b9qx3l6qod0ti1kg.eu-west-1.aws.endpoints.huggingface.cloud"
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1024 if target_lang in ("explain", "summarize") else 512,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            }
            print(f"  [ONLINE MODEL] Calling dedicated qwen3 endpoint...")
        else:
            url = "https://router.huggingface.co/v1/chat/completions"
            payload = {
                "model": "minhatou/qwen2",
                "messages": messages,
                "max_tokens": 1024 if target_lang in ("explain", "summarize") else 512,
                "temperature": 0.1
            }
            print(f"  [ONLINE MODEL] Calling minhatou/qwen2 via router.huggingface.co chat completions...")
        
        detected_src = detect_language(text)
        print(f"  [ONLINE MODEL] detected_src={detected_src} → target={target_lang}")
        
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            elapsed = time.time() - start_time
            print(f"  [ONLINE MODEL] HTTP status code: {response.status_code} (took {elapsed:.2f} seconds)")
            
            if response.status_code != 200:
                print(f"  [ONLINE MODEL ERROR] Raw API Response: {response.text}")
                raise Exception(f"Hugging Face API returned error status {response.status_code}: {response.text}")
            
            res_data = response.json()
            if model_id == "qwen3":
                if isinstance(res_data, list) and len(res_data) > 0:
                    result = res_data[0].get("generated_text", "").strip()
                elif isinstance(res_data, dict):
                    result = res_data.get("generated_text", "").strip()
                else:
                    raise Exception(f"Unexpected response format from HF Endpoint: {res_data}")
                
                # Robustly strip prompt using the assistant turn token
                if "<|im_start|>assistant" in result:
                    result = result.split("<|im_start|>assistant")[-1].strip()
                    if result.startswith(":"):
                        result = result[1:].strip()
                elif result.startswith(prompt):
                    result = result[len(prompt):].strip()
            else:
                result = res_data["choices"][0]["message"]["content"].strip()
            print(f"  [ONLINE MODEL] Successful response. Output length: {len(result)} chars.")
        except Exception as e:
            import traceback
            print("  [ONLINE MODEL EXCEPTION] Traceback:")
            traceback.print_exc()
            raise e
    else:
        # Local model inference using PyTorch and Transformers
        import torch
        start_time = time.time()
        print(f"  [LOCAL MODEL] detected_src={detect_language(text)} → target={target_lang}")
        print(f"  [LOCAL MODEL] Running local inference via GPU...")
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            max_new = 1024 if target_lang in ("explain", "summarize") else 512
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # Decode generated output (excluding the input prompt tokens)
            input_len = inputs.input_ids.shape[1]
            result = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
            elapsed = time.time() - start_time
            print(f"  [LOCAL MODEL] Inference successful (took {elapsed:.2f} seconds). Output length: {len(result)} chars.")
        except Exception as e:
            import traceback
            print("  [LOCAL MODEL EXCEPTION] Traceback:")
            traceback.print_exc()
            raise e

    # Clean up reasoning <think> tags or thoughts
    if "<think>" in result and "</think>" in result:
        import re
        result = re.sub(r'<think>[\s\S]*?</think>', '', result)
    else:
        result = result.replace("<think>", "").replace("</think>", "")
    result = result.strip()

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

