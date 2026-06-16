import os
import requests

# Vietnamese diacritic characters for language detection
VI_CHARS = set(
    'àáảãạăắặẳẵâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
    'ÀÁẢÃẠĂẮẶẲẴÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ'
)

# Filler phrases small models tend to append after the translation
FILLER_MARKERS = [
    "Note:", "note:", "Here is", "here is",
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


def generate_translation(model, tokenizer, text, context="", target_lang="auto", glossary=None, glossary_mode="both", model_id="qwen3", model_path=None):
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

    # Rule B: Prompt Context Enforced (only for non-explain and non-summarize modes)
    glossary_context = ""
    if target_lang not in ("explain", "summarize") and glossary and glossary_mode in ("both", "ai"):
        detected_glossary = {}
        text_lower = text.lower()
        for k, v in glossary.items():
            k_clean = str(k).strip().lower()
            if f" {k_clean} " in f" {text_lower} " or text_lower.startswith(k_clean) or text_lower.endswith(k_clean):
                detected_glossary[str(k).strip()] = str(v).strip()
        
        if detected_glossary:
            import json as _json
            glossary_context = _json.dumps(detected_glossary, ensure_ascii=False)
            print(f"  [GLOSSARY] Enforced glossary prompt context (JSON): {glossary_context}")

    if target_lang == "vietnamese":
        system_prompt = (
            "Bạn là một biên dịch viên chuyên nghiệp về công nghệ thông tin. "
            "Nhiệm vụ của bạn là dịch đoạn văn bản nằm trong khối [TEXT_TO_TRANSLATE] sang tiếng Việt. "
            "Khối [CONTEXT/GLOSSARY] chứa các thông tin ngữ cảnh hoặc các cặp từ gợi ý dạng JSON. "
            "Nếu có các cặp thuật ngữ được định nghĩa trong [CONTEXT/GLOSSARY], hãy bắt buộc ưu tiên sử dụng "
            "chúng để dịch các từ tương ứng trong [TEXT_TO_TRANSLATE]. "
            "Tuyệt đối KHÔNG được dịch các câu trong khối [CONTEXT/GLOSSARY] hay đưa bất kỳ nội dung nào khác ngoài "
            "bản dịch trực tiếp của [TEXT_TO_TRANSLATE] vào kết quả. "
            "Hãy CHỈ trả về bản dịch trực tiếp, KHÔNG giải thích, không thêm tiêu đề hay từ ngữ thừa nào khác."
        )
    elif target_lang == "english":
        lang_instruction = (
            "Translate the given text from Vietnamese into English. "
            "Use accurate IT-specific English terminology when the context is technical."
        )
        system_prompt = (
            "Bạn là một biên dịch viên chuyên nghiệp về công nghệ thông tin. "
            "Nhiệm vụ cốt lõi của bạn là CHỈ dịch đoạn văn bản nằm trong khối [TEXT_TO_TRANSLATE] sang tiếng Anh. "
            "Khối [CONTEXT] chỉ nhằm mục đích cung cấp ngữ cảnh, tuyệt đối KHÔNG dịch bất kỳ câu nào trong khối [CONTEXT]. "
            "Quy tắc nghiêm ngặt: Hãy so sánh kỹ hai khối [TEXT_TO_TRANSLATE] và [CONTEXT]. Bạn chỉ được dịch câu chữ xuất hiện trong [TEXT_TO_TRANSLATE]. "
            "Mọi câu khác xuất hiện trong [CONTEXT] nhưng không có trong [TEXT_TO_TRANSLATE] thì TUYỆT ĐỐI KHÔNG DỊCH."
        )
        if glossary_context:
            system_prompt += (
                "\nKhi dịch, nếu gặp các từ khóa sau, bạn bắt buộc phải dịch chúng theo đúng định nghĩa này:\n"
                f"{glossary_context}\n"
            )
        system_prompt += "\nHãy CHỈ trả về bản dịch trực tiếp của văn bản trong khối [TEXT_TO_TRANSLATE], không dịch khối [CONTEXT], không giải thích, không thêm tiêu đề, nhãn hay từ ngữ thừa nào khác."
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
            "Chỉ liệt kê tối đa 3-4 gạch đầu dòng cốt lỗi, mỗi gạch đầu dòng chỉ gồm 1 câu cực kỳ ngắn gọn, tập trung vào thông tin quan trọng nhất."
        )
    
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
            f"[CONTEXT/GLOSSARY]\n{glossary_context if glossary_context else context}\n[/CONTEXT/GLOSSARY]\n\n"
            f"[TEXT_TO_TRANSLATE]\n{text}\n[/TEXT_TO_TRANSLATE]\n\n"
            f"Instruction: Dịch đoạn văn bản trên sang tiếng Việt. Ưu tiên sử dụng các cặp thuật ngữ IT được định nghĩa trong khối CONTEXT/GLOSSARY nếu có.\nResult:"
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
            
        is_chat_completions_endpoint = False
        is_dedicated_endpoint = False
        if model_path:
            if model_path.startswith("http://") or model_path.startswith("https://"):
                url = model_path
                is_dedicated_endpoint = True
                if url.endswith("/v1/chat/completions") or url.endswith("/chat/completions"):
                    is_chat_completions_endpoint = True
                else:
                    is_chat_completions_endpoint = False
            else:
                url = "https://router.huggingface.co/v1/chat/completions"
                is_chat_completions_endpoint = True
                is_dedicated_endpoint = False
        else:
            if model_id == "qwen3":
                url = "https://b9qx3l6qod0ti1kg.eu-west-1.aws.endpoints.huggingface.cloud"
                is_chat_completions_endpoint = False
                is_dedicated_endpoint = True
            else:
                url = "https://router.huggingface.co/v1/chat/completions"
                is_chat_completions_endpoint = True
                is_dedicated_endpoint = False

        if not is_chat_completions_endpoint:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1024 if target_lang in ("explain", "summarize") else 512,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            }
            print(f"  [ONLINE MODEL] Calling TGI endpoint: {url}")
        else:
            if "/" in model_id:
                resolved_model_name = model_id
            else:
                resolved_model_name = model_path if (model_path and not model_path.startswith("http")) else f"minhatou/{model_id.lower()}"
                
            if resolved_model_name in ("minhatou/qwen2-3b", "tgi"):
                resolved_model_name = "minhatou/qwen2.5_3b_1106"
                
            payload = {
                "model": resolved_model_name,
                "messages": messages,
                "max_tokens": 1024 if target_lang in ("explain", "summarize") else 512,
                "temperature": 0.1
            }
            print(f"  [ONLINE MODEL] Calling chat completions endpoint ({resolved_model_name}) via {url}...")
            
        print(f"  [ONLINE MODEL REQUEST DETAIL] URL: {url}")
        print(f"  [ONLINE MODEL REQUEST DETAIL] Headers: {{k: ('***' if k.lower() == 'authorization' else v) for k, v in headers.items()}}")
        print(f"  [ONLINE MODEL REQUEST DETAIL] Payload: {payload}")
        
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            elapsed = time.time() - start_time
            print(f"  [ONLINE MODEL] HTTP status code: {response.status_code} (took {elapsed:.2f} seconds)")
            
            if response.status_code != 200:
                print(f"  [ONLINE MODEL ERROR] Raw API Response: {response.text}")
                raise Exception(f"Hugging Face API returned error status {response.status_code}: {response.text}")
            
            res_data = response.json()
            if not is_chat_completions_endpoint:
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
    
    # Strip any ending patterns like (Text_to_translate) or [TEXT_TO_TRANSLATE] case-insensitively
    import re
    result = re.sub(r'[\(\[\{]\s*(text_to_translate|context|translation)\s*[\)\]\}]', '', result, flags=re.IGNORECASE)
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
        # Remove any structural tag lines copied/mimicked by the model (e.g. [CONTEXT], [/CONTEXT], [TEXT_TO_TRANSLATE])
        filtered_lines = [l for l in lines if not (l.startswith("[") and l.endswith("]"))]
        return filtered_lines[0] if filtered_lines else result
        
    return result

