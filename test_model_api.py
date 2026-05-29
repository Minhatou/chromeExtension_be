import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)  # Enable CORS for easy testing from browsers or Postman

# Load HF Token
def get_hf_token():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        return line.strip().split("=", 1)[1]
        except Exception:
            pass
    return hf_token or ""

HF_TOKEN = get_hf_token()

@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Expose a clean endpoint to test your minhatou/qwen2 model with any prompt.
    Expects JSON body:
    {
        "prompt": "Văn bản cần gửi hoặc câu hỏi cho mô hình",
        "system_prompt": "Hướng dẫn hệ thống (Tùy chọn)",
        "temperature": 0.1,
        "max_new_tokens": 512
    }
    """
    data = request.json or {}
    user_prompt = data.get("prompt")
    
    if not user_prompt:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400
        
    system_prompt = data.get("system_prompt", "Bạn là trợ lý AI hữu ích.")
    temperature = data.get("temperature", 0.1)
    max_new_tokens = data.get("max_new_tokens", 512)
    
    # Render with Qwen2 chat template format
    formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\n[TEST API REQUEST]")
    print(f"  User Prompt: {repr(user_prompt)}")
    print(f"  System Prompt: {repr(system_prompt)}")
    
    start_time = time.time()
    try:
        import requests
        url = "https://api-inference.huggingface.co/models/minhatou/qwen2"
        headers = {"Content-Type": "application/json"}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise Exception(f"HF API returned {response.status_code}: {response.text}")
            
        res_data = response.json()
        if isinstance(res_data, list) and len(res_data) > 0 and "generated_text" in res_data[0]:
            response_text = res_data[0]["generated_text"].strip()
        else:
            raise Exception(f"Unexpected response format: {res_data}")
        elapsed = time.time() - start_time
        print(f"  [SUCCESS] Generated in {elapsed:.2f}s, length: {len(response_text)} chars.")
        
        return jsonify({
            "status": "success",
            "model": "minhatou/qwen2",
            "elapsed_seconds": round(elapsed, 2),
            "generated_text": response_text.strip()
        })
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        return jsonify({
            "status": "error",
            "error_message": str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "running",
        "model": "minhatou/qwen2",
        "token_loaded": len(HF_TOKEN) > 0
    })

if __name__ == '__main__':
    print("==========================================================")
    print("Starting Standalone Test API for minhatou/qwen2...")
    print("Access endpoints at http://127.0.0.1:5001")
    print("Endpoints:")
    print("  GET  /api/status   - Check status")
    print("  POST /api/generate - Test prompt generation")
    print("==========================================================")
    app.run(host='0.0.0.0', port=5001, debug=False)
