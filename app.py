from flask import Flask, request, jsonify
from flask_cors import CORS
from model_inference import load_model, generate_translation
import threading

app = Flask(__name__)
CORS(app)

# Global variables to hold model and tokenizer
model = None
tokenizer = None

# Thread lock to prevent concurrent VRAM usage and OOM errors during inference
inference_lock = threading.Lock()

@app.route('/api/status', methods=['GET'])
def status():
    """Check if the API and model are running."""
    return jsonify({
        "status": "running", 
        "model_loaded": model is not None
    })

@app.route('/api/translate', methods=['POST'])
def translate():
    """Endpoint to process context-aware translation."""
    global model, tokenizer
    
    # In a real scenario, we might want to load the model lazily
    # or ensure it's loaded before handling requests.
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    text = data['text']
    context = data.get('context', '')
    
    # Use a lock to ensure only one inference happens at a time preventing VRAM OOM
    with inference_lock:
        try:
            translation = generate_translation(model, tokenizer, text, context)
            return jsonify({"translation": translation})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # ── Model config ──────────────────────────────────────────────────────
    BASE_MODEL_PATH   = "./models/llama-3.2-1b-instruct"
    # Set to adapter path after training, or None to use the base model only:
    LORA_WEIGHTS_PATH = None
    # LORA_WEIGHTS_PATH = "./models/lora-adapter"
    # ─────────────────────────────────────────────────────────────────────

    print("Loading model, please wait...")
    model, tokenizer = load_model(
        base_model_name=BASE_MODEL_PATH,
        lora_weights_path=LORA_WEIGHTS_PATH,
    )
    print("Model loaded. Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)

