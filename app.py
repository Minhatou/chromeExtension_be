from flask import Flask, request, jsonify
from flask_cors import CORS
from model_inference import load_model, generate_translation
import threading

# ── Firebase Admin SDK ────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore, auth as fb_auth

def init_ai_models():
    """Ensure default models are loaded into the AI_model collection in Firestore."""
    if db is None:
        return
    try:
        models_ref = db.collection("AI_model")
        docs = models_ref.limit(1).get()
        if len(docs) == 0:
            print("[FIREBASE] Initializing default AI_model entries...")
            models_ref.document("qwen2").set({
                "model_id": "qwen2",
                "name": "Qwen2-1.5b",
                "path": r"C:\Users\Cko Ckeems Ngoo\LlamaFactory\qwen_7278",
                "input_price_1m": 5000.0,
                "output_price_1m": 15000.0
            })
            models_ref.document("qwen3").set({
                "model_id": "qwen3",
                "name": "Qwen3-1.7b",
                "path": r"C:\Users\Cko Ckeems Ngoo\LlamaFactory\qwen3-1.7b-7278",
                "input_price_1m": 7000.0,
                "output_price_1m": 21000.0
            })
            print("[FIREBASE] Default AI_model entries initialized successfully!")
    except Exception as e:
        print(f"[FIREBASE ERROR] Failed to initialize default AI_model entries: {e}")

try:
    import json
    import os
    service_account_env = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if service_account_env:
        service_account_info = json.loads(service_account_env)
        cred = credentials.Certificate(service_account_info)
        print("[FIREBASE] Initialized Firebase Admin SDK using Environment Variable")
    else:
        cred = credentials.Certificate("serviceAccountKey.json")
        print("[FIREBASE] Initialized Firebase Admin SDK using local file")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("[FIREBASE] Connected to Cloud Firestore!")
    init_ai_models()
except Exception as e:
    print(f"[FIREBASE ERROR] {e}")
    db = None
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

from payos import PayOS
from payos.types import CreatePaymentLinkRequest, ItemData
payos_client = None
try:
    payos_client = PayOS(
        client_id=os.environ.get("PAYOS_CLIENT_ID"),
        api_key=os.environ.get("PAYOS_API_KEY"),
        checksum_key=os.environ.get("PAYOS_CHECKSUM_KEY")
    )
    print("[PAYOS] Initialized successfully!")
except Exception as e:
    print(f"[PAYOS ERROR] Failed to initialize PayOS: {e}")


@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        res = app.make_response('')
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        res.status_code = 200
        return res

# Global dictionary to hold models and tokenizers dynamically
loaded_models = {}
loaded_tokenizers = {}

# Thread lock to prevent concurrent VRAM usage and OOM errors during inference
inference_lock = threading.Lock()

def get_model_and_tokenizer(model_id):
    """Dynamic thread-safe loader for Qwen2 and Qwen3 models."""
    global loaded_models, loaded_tokenizers
    
    path = None
    if db is not None:
        try:
            doc = db.collection("AI_model").document(model_id).get()
            if doc.exists:
                path = doc.to_dict().get("path")
        except Exception as e:
            print(f"[FIREBASE ERROR] Failed to fetch path for model {model_id}: {e}")
            
    if not path:
        model_paths = {
            "qwen2": r"C:\Users\Cko Ckeems Ngoo\LlamaFactory\qwen_7278",
            "qwen3": r"C:\Users\Cko Ckeems Ngoo\LlamaFactory\qwen3-1.7b-7278"
        }
        path = model_paths.get(model_id, model_paths["qwen3"])
    
    if model_id not in loaded_models:
        print(f"[DYNAMIC LOAD] Loading model {model_id} from: {path}...")
        m, t = load_model(base_model_name=path, lora_weights_path=None)
        loaded_models[model_id] = m
        loaded_tokenizers[model_id] = t
        print(f"[DYNAMIC LOAD] Model {model_id} loaded successfully!")
        
    return loaded_models[model_id], loaded_tokenizers[model_id], path


# ── Firebase Helper Functions ─────────────────────────────────────────────────

def get_or_init_user_credits(user_id):
    """Get or initialize user's credits (VND) in Firestore. Defaults to 100,000đ free credit per day."""
    if db is None or not user_id or user_id == "anonymous":
        return {"free_credit": 0.0, "purchased_credit": 0.0, "total_credit": 0.0}
    try:
        import datetime
        tz_delta = datetime.timezone(datetime.timedelta(hours=7)) # Vietnam time
        now_vn = datetime.datetime.now(tz_delta)
        today_str = now_vn.strftime("%Y-%m-%d")

        user_ref = db.collection("user_info").document(user_id)
        doc = user_ref.get()
        
        free_credit = 100000.0
        purchased_credit = 0.0
        last_reset_date = today_str
        theme = "light"
        share_translation = False
        
        if doc.exists:
            data = doc.to_dict()
            free_credit = float(data.get("free_credit", 100000.0))
            purchased_credit = float(data.get("purchased_credit", 0.0))
            last_reset_date = data.get("last_credit_reset_date", "")
            theme = data.get("theme", "light")
            share_translation = data.get("share_translation", False)
            
            # Check for daily reset
            if last_reset_date != today_str:
                print(f"  [FIREBASE] Daily reset triggered for {user_id}. {last_reset_date} -> {today_str}")
                free_credit = 100000.0
                last_reset_date = today_str
                user_ref.set({
                    "free_credit": free_credit,
                    "last_credit_reset_date": today_str
                }, merge=True)
        else:
            # Create user_info document
            user_ref.set({
                "free_credit": free_credit,
                "purchased_credit": purchased_credit,
                "last_credit_reset_date": today_str,
                "theme": theme,
                "share_translation": share_translation
            })
            print(f"  [FIREBASE] Initialized user credits for: {user_id}")
            
        return {
            "free_credit": free_credit,
            "purchased_credit": purchased_credit,
            "total_credit": free_credit + purchased_credit,
            "theme": theme,
            "share_translation": share_translation
        }
    except Exception as e:
        print(f"  [FIREBASE ERROR] Failed to load/reset user credits: {e}")
        return {"free_credit": 100000.0, "purchased_credit": 0.0, "total_credit": 100000.0, "theme": "light", "share_translation": False}

def deduct_user_credits(user_id, amount_vnd):
    """Deduct credit (VND) from user's balance in Firestore, prioritizing free credit first."""
    if db is None or not user_id or user_id == "anonymous":
        return {"free_credit": 0.0, "purchased_credit": 0.0, "total_credit": 0.0}
    try:
        user_ref = db.collection("user_info").document(user_id)
        balances = get_or_init_user_credits(user_id)
        
        free = balances["free_credit"]
        purchased = balances["purchased_credit"]
        
        if free >= amount_vnd:
            free = max(0.0, free - amount_vnd)
        else:
            remainder = amount_vnd - free
            free = 0.0
            purchased = max(0.0, purchased - remainder)
            
        user_ref.set({
            "free_credit": free,
            "purchased_credit": purchased
        }, merge=True)
        
        total = free + purchased
        print(f"  [FIREBASE] Deducted {amount_vnd:.2f} VND from: {user_id}. New balance: {total:.2f} VND")
        return {
            "free_credit": free,
            "purchased_credit": purchased,
            "total_credit": total
        }
    except Exception as e:
        print(f"  [FIREBASE ERROR] Failed to deduct credits: {e}")
        return {"free_credit": 0.0, "purchased_credit": 0.0, "total_credit": 0.0}

def get_user_glossary(user_id):
    """Load user's personal glossary from Firestore."""
    if db is None or not user_id or user_id == "anonymous":
        return {}
    try:
        glossary = {}
        docs = db.collection("users").document(user_id).collection("glossary").stream()
        for doc in docs:
            data = doc.to_dict()
            if "term" in data and "definition" in data:
                glossary[data["term"].strip().lower()] = data["definition"].strip()
        print(f"  [FIREBASE] Loaded {len(glossary)} glossary entries for: {user_id}")
        return glossary
    except Exception as e:
        print(f"  [FIREBASE ERROR] Failed to load glossary: {e}")
        return {}


def save_user_private_history(user_id, source, target):
    """Save translation entry to user's private history in Firestore."""
    if db is None or not user_id or user_id == "anonymous":
        return
    try:
        db.collection("users").document(user_id).collection("history").add({
            "source": source.strip(),
            "target": target.strip(),
            "time": "Vừa xong",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"  [FIREBASE ERROR] Failed to save private user history: {e}")

def save_translation_log(source, result, task_type, model_id):
    """Save translation log to Firestore asynchronously and anonymously."""
    if db is None:
        return
    try:
        db.collection("translation_logs").add({
            "source_text": source,
            "translated_text": result,
            "type": task_type,
            "model_id": model_id,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print(f"  [FIREBASE] Saved anonymous translation log to translation_logs successfully!")
    except Exception as e:
        print(f"  [FIREBASE ERROR] Failed to save anonymous log: {e}")

def check_translation_cache(user_id, text):
    """Check if a translation already exists in saved_translations or history (with no dislike)."""
    if db is None or not user_id or user_id == "anonymous":
        return None
    try:
        # 1. Check saved_translations
        docs = db.collection("users").document(user_id).collection("saved_translations")\
                 .where("source_text", "==", text.strip()).limit(1).stream()
        for doc in docs:
            data = doc.to_dict()
            if "translated_text" in data:
                print(f"  [CACHE HIT] Found translation in saved_translations: {data['translated_text']}")
                return data["translated_text"]

        # 2. Check history
        docs = db.collection("users").document(user_id).collection("history")\
                 .where("source", "==", text.strip()).stream()
        latest_translation = None
        latest_ts = None
        for doc in docs:
            data = doc.to_dict()
            if data.get("rating") == "dislike":
                print(f"  [CACHE BYPASS] Found a disliked translation of this text. Bypassing cache.")
                return None
            ts = data.get("timestamp")
            if "target" in data:
                if latest_ts is None or (ts and ts > latest_ts):
                    latest_ts = ts
                    latest_translation = data["target"]
        
        if latest_translation:
            print(f"  [CACHE HIT] Found translation in history: {latest_translation}")
        return latest_translation
    except Exception as e:
        print(f"  [CACHE ERROR] Failed to lookup cache: {e}")
        return None

# ── Glossary Routes ───────────────────────────────────────────────────────────

@app.route('/api/glossary', methods=['POST'])
def get_glossary():
    """Fetch user's glossary from Firestore."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    uid = request.json.get('uid')
    if not uid:
        return jsonify({"error": "Missing uid"}), 400
    try:
        docs = db.collection("users").document(uid).collection("glossary").stream()
        glossary_list = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            # Frontend uses 'meaning', backend uses 'definition'
            if 'definition' in data and 'meaning' not in data:
                data['meaning'] = data['definition']
            glossary_list.append(data)
        return jsonify({"glossary": glossary_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/glossary/add', methods=['POST'])
def add_glossary_term():
    """Add a term to user's glossary."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    data = request.json
    uid = data.get('uid')
    term = data.get('term')
    meaning = data.get('meaning')
    context = data.get('context', 'Thêm thủ công')

    if not uid or not term or not meaning:
        return jsonify({"error": "Missing required fields"}), 400
    
    new_entry = {
        "term": term,
        "definition": meaning,  # for backend usage
        "meaning": meaning,     # for frontend usage
        "context": context,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    try:
        _, doc_ref = db.collection("users").document(uid).collection("glossary").add(new_entry)
        new_entry['id'] = doc_ref.id
        new_entry['timestamp'] = None # Not serializable directly
        return jsonify({"success": True, "entry": new_entry})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/glossary/delete', methods=['POST'])
def delete_glossary_term():
    """Delete a term from user's glossary."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    data = request.json
    uid = data.get('uid')
    doc_id = data.get('id')

    if not uid or not doc_id:
        return jsonify({"error": "Missing uid or id"}), 400
    try:
        db.collection("users").document(uid).collection("glossary").document(doc_id).delete()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── History Routes ────────────────────────────────────────────────────────────

@app.route('/api/history', methods=['POST'])
def get_history():
    """Fetch user's translation history from Firestore."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    uid = request.json.get('uid')
    if not uid:
        return jsonify({"error": "Missing uid"}), 400
    try:
        # Get history ordered by timestamp descending, limit to 50
        docs = db.collection("users").document(uid).collection("history")\
                 .order_by("timestamp", direction=firestore.Query.DESCENDING).limit(50).stream()
        history_list = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            # Frontend uses 'time', 'source', 'target'
            history_list.append(data)
        return jsonify({"history": history_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history/add', methods=['POST'])
def add_history_entry():
    """Add a translation entry to user's history."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    data = request.json
    uid = data.get('uid')
    source = data.get('source')
    target = data.get('target')
    time = data.get('time', 'Vừa xong')

    if not uid or not source or not target:
        return jsonify({"error": "Missing required fields"}), 400
    
    new_entry = {
        "source": source,
        "target": target,
        "time": time,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    try:
        _, doc_ref = db.collection("users").document(uid).collection("history").add(new_entry)
        new_entry['id'] = doc_ref.id
        new_entry['timestamp'] = None
        return jsonify({"success": True, "entry": new_entry})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear all translation history for user."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    uid = request.json.get('uid')
    if not uid:
        return jsonify({"error": "Missing uid"}), 400
    try:
        # Delete all documents in history collection (Note: this is a simple approach, for large collections use batches)
        docs = db.collection("users").document(uid).collection("history").stream()
        for doc in docs:
            doc.reference.delete()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Auth Routes ──────────────────────────────────────────────────────────────

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login endpoint - Extension sends email/password, Flask verifies via Firebase REST API."""
    data = request.json
    email = data.get('email', '')
    password = data.get('password', '')

    if not email or not password:
        return jsonify({"error": "Email và mật khẩu không được để trống"}), 400

    try:
        # Load API key from env or .env file manually without requiring python-dotenv
        import os
        api_key = os.environ.get('FIREBASE_API_KEY')
        if not api_key:
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("FIREBASE_API_KEY="):
                            api_key = line.strip().split("=", 1)[1]
            except Exception:
                pass
                
        if not api_key:
            return jsonify({"error": "Server is missing FIREBASE_API_KEY configuration."}), 500

        # Verify via Firebase REST API using standard urllib
        import urllib.request
        import json
        
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        req = urllib.request.Request(url, method="POST", headers={"Content-Type": "application/json"})
        body = json.dumps({"email": email, "password": password, "returnSecureToken": True}).encode("utf-8")
        
        try:
            with urllib.request.urlopen(req, data=body) as response:
                res_data = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            print(f"[FIREBASE LOGIN ERROR] HTTP {e.code}: {err_body}")
            return jsonify({"error": f"Đăng nhập thất bại (HTTP {e.code})", "details": err_body}), 401

        uid = res_data.get("localId")
        id_token = res_data.get("idToken")
        refresh_token = res_data.get("refreshToken")
        
        # Get custom claims (role) using Admin SDK
        user = fb_auth.get_user(uid)
        role = user.custom_claims.get("role", "user") if user.custom_claims else "user"
        
        if role == "admin":
            credits = {"free_credit": -1.0, "purchased_credit": -1.0, "total_credit": -1.0}
        else:
            credits = get_or_init_user_credits(uid)
        
        return jsonify({
            "uid": uid,
            "email": email,
            "role": role,
            "idToken": id_token,
            "refreshToken": refresh_token,
            "free_credit": credits["free_credit"],
            "purchased_credit": credits["purchased_credit"],
            "total_credit": credits["total_credit"],
            "theme": credits.get("theme", "light") if role != "admin" else "light"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register endpoint - Extension sends email/password to create account."""
    data = request.json
    email = data.get('email', '')
    password = data.get('password', '')

    if not email or not password:
        return jsonify({"error": "Email và mật khẩu không được để trống"}), 400

    try:
        import os
        api_key = os.environ.get('FIREBASE_API_KEY')
        if not api_key:
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("FIREBASE_API_KEY="):
                            api_key = line.strip().split("=", 1)[1]
            except Exception:
                pass
                
        if not api_key:
            return jsonify({"error": "Server is missing FIREBASE_API_KEY configuration."}), 500

        import urllib.request
        import json
        
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
        req = urllib.request.Request(url, method="POST", headers={"Content-Type": "application/json"})
        body = json.dumps({"email": email, "password": password, "returnSecureToken": True}).encode("utf-8")
        
        try:
            with urllib.request.urlopen(req, data=body) as response:
                res_data = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            print(f"[FIREBASE REGISTER ERROR] HTTP {e.code}: {err_body}")
            if "EMAIL_EXISTS" in err_body:
                return jsonify({"error": "Email này đã được đăng ký"}), 400
            if "WEAK_PASSWORD" in err_body:
                return jsonify({"error": "Mật khẩu phải có ít nhất 6 ký tự"}), 400
            return jsonify({"error": f"Đăng ký thất bại (HTTP {e.code})", "details": err_body}), 400

        uid = res_data.get("localId")
        id_token = res_data.get("idToken")
        refresh_token = res_data.get("refreshToken")
        
        credits = get_or_init_user_credits(uid)
        
        return jsonify({
            "uid": uid,
            "email": email,
            "role": "user",
            "idToken": id_token,
            "refreshToken": refresh_token,
            "free_credit": credits["free_credit"],
            "purchased_credit": credits["purchased_credit"],
            "total_credit": credits["total_credit"],
            "theme": credits.get("theme", "light")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/auth/google')
def google_auth_page():
    """Serves a simple HTML page that uses Firebase Web SDK from CDN to do Google Sign-In with popup."""
    import os
    api_key = os.environ.get('FIREBASE_API_KEY')
    if not api_key:
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("FIREBASE_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
        except Exception:
            pass
            
    if not api_key:
        api_key = "AIzaSyCjfMHgbOzSe7HUoebI2u2xvJVq_NY6aws"
        
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Google Sign-In - IT Translator</title>
    <script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-auth-compat.js"></script>
    <style>
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: #121214;
            color: #ffffff;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            text-align: center;
        }}
        .spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.1);
            border-left-color: #3b82f6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="spinner"></div>
    <h3>Đang kết nối tài khoản Google...</h3>
    <p style="color: #888; font-size: 14px;">Vui lòng hoàn tất đăng nhập ở cửa sổ popup tiếp theo.</p>

    <script>
        const firebaseConfig = {{
            apiKey: "{api_key}",
            authDomain: "doan-4ee1f.firebaseapp.com"
        }};
        firebase.initializeApp(firebaseConfig);

        const provider = new firebase.auth.GoogleAuthProvider();
        firebase.auth().signInWithPopup(provider)
            .then(async (result) => {{
                const idToken = await result.user.getIdToken();
                const refreshToken = result.user.refreshToken;
                window.opener.postMessage({{ type: 'GOOGLE_AUTH_SUCCESS', idToken: idToken, refreshToken: refreshToken }}, '*');
                document.body.innerHTML = '<h3 style="color: #4cd137;">Đăng nhập thành công!</h3><p>Đang quay lại ứng dụng...</p>';
                setTimeout(() => window.close(), 1000);
            }})
            .catch((error) => {{
                window.opener.postMessage({{ type: 'GOOGLE_AUTH_ERROR', error: error.message }}, '*');
                document.body.innerHTML = '<h3 style="color: #ff7675;">Đăng nhập thất bại</h3><p>' + error.message + '</p>';
                setTimeout(() => window.close(), 3000);
            }});
    </script>
</body>
</html>
"""

@app.route('/api/auth/google', methods=['POST'])
def google_login():
    """Verify Firebase ID token obtained from Google Sign-In on client side."""
    data = request.json
    id_token = data.get('idToken', '')
    refresh_token = data.get('refreshToken', '')
    if not id_token:
        return jsonify({"error": "Missing idToken"}), 400
    try:
        decoded = fb_auth.verify_id_token(id_token)
        uid = decoded.get("uid")
        email = decoded.get("email", "")
        
        user = fb_auth.get_user(uid)
        role = user.custom_claims.get("role", "user") if user.custom_claims and user.custom_claims.get("role") else "user"
        
        if role == "admin":
            credits = {"free_credit": -1.0, "purchased_credit": -1.0, "total_credit": -1.0}
        else:
            credits = get_or_init_user_credits(uid)
            
        return jsonify({
            "uid": uid,
            "email": email,
            "role": role,
            "idToken": id_token,
            "refreshToken": refresh_token,
            "free_credit": credits["free_credit"],
            "purchased_credit": credits["purchased_credit"],
            "total_credit": credits["total_credit"],
            "theme": credits.get("theme", "light") if role != "admin" else "light"
        })
    except Exception as e:
        print(f"[FIREBASE GOOGLE LOGIN ERROR] {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/refresh', methods=['POST'])
def refresh_token():
    """Exchange a refresh token for a new ID token."""
    data = request.json
    r_token = data.get('refreshToken') if data else None
    if not r_token:
        return jsonify({"error": "Missing refreshToken"}), 400
    try:
        import os
        api_key = os.environ.get('FIREBASE_API_KEY')
        if not api_key:
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("FIREBASE_API_KEY="):
                            api_key = line.strip().split("=", 1)[1]
            except Exception:
                pass
        if not api_key:
            return jsonify({"error": "Server is missing FIREBASE_API_KEY configuration."}), 500

        import urllib.request
        import json
        
        url = f"https://securetoken.googleapis.com/v1/token?key={api_key}"
        req = urllib.request.Request(url, method="POST", headers={"Content-Type": "application/x-www-form-urlencoded"})
        body = f"grant_type=refresh_token&refresh_token={r_token}".encode("utf-8")
        
        try:
            with urllib.request.urlopen(req, data=body) as response:
                res_data = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            print(f"[FIREBASE REFRESH ERROR] HTTP {e.code}: {err_body}")
            return jsonify({"error": "Failed to refresh token", "details": err_body}), 401
            
        return jsonify({
            "idToken": res_data.get("id_token"),
            "refreshToken": res_data.get("refresh_token")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/verify', methods=['POST'])
def verify_token():
    """Verify Firebase ID token sent from Chrome Extension. Returns uid, email, role."""
    data = request.json
    id_token = data.get('idToken', '') if data else ''
    if not id_token:
        return jsonify({"valid": False, "error": "No token provided"}), 400
    try:
        decoded = fb_auth.verify_id_token(id_token)
        uid   = decoded.get("uid")
        email = decoded.get("email", "")
        role  = decoded.get("role", "user")   # Custom claim set by create_admin.py
        
        if role == "admin":
            credits = {"free_credit": -1.0, "purchased_credit": -1.0, "total_credit": -1.0}
        else:
            credits = get_or_init_user_credits(uid)
        
        print(f"[AUTH] Token verified: uid={uid} email={email} role={role} free={credits['free_credit']} purchased={credits['purchased_credit']}")
        return jsonify({
            "valid": True,
            "uid": uid,
            "email": email,
            "role": role,
            "free_credit": credits["free_credit"],
            "purchased_credit": credits["purchased_credit"],
            "total_credit": credits["total_credit"],
            "theme": credits.get("theme", "light") if role != "admin" else "light",
            "share_translation": credits.get("share_translation", False) if role != "admin" else False
        })
    except fb_auth.ExpiredIdTokenError:
        return jsonify({"valid": False, "error": "Token expired"}), 401
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 401


@app.route('/api/user/theme', methods=['POST'])
def update_user_theme():
    """Update user's preferred theme (light/dark) in Firestore."""
    data = request.json
    uid = data.get('uid')
    theme = data.get('theme', 'light')
    
    if not uid:
        return jsonify({"error": "Missing uid"}), 400
        
    try:
        user_ref = db.collection("user_info").document(uid)
        user_ref.set({"theme": theme}, merge=True)
        return jsonify({"success": True, "theme": theme})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/user/share_translation', methods=['POST'])
def update_user_share_translation():
    """Update user's share translation preference in Firestore."""
    data = request.json
    uid = data.get('uid')
    share = data.get('share_translation', False)
    
    if not uid:
        return jsonify({"error": "Missing uid"}), 400
        
    try:
        user_ref = db.collection("user_info").document(uid)
        user_ref.set({"share_translation": share}, merge=True)
        return jsonify({"success": True, "share_translation": share})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/user/recharge', methods=['POST'])
def recharge_tokens():
    """Mock recharge credit (VND) for users. Supports packages: basic, standard, premium."""
    data = request.json
    uid = data.get('uid')
    package_id = data.get('package_id')  # 'basic' | 'standard' | 'premium'
    payment_method = data.get('payment_method', 'qr')

    if not uid:
        return jsonify({"error": "Missing uid"}), 400
    if not package_id:
        return jsonify({"error": "Missing package_id"}), 400

    packages = {
        "basic": 50000.0,
        "standard": 200000.0,
        "premium": 500000.0
    }

    if package_id == "custom":
        amount = data.get("amount")
        try:
            added_credit = float(amount)
        except (TypeError, ValueError):
            return jsonify({"error": "Số tiền nạp không hợp lệ"}), 400
        if added_credit < 1000.0:
            return jsonify({"error": "Số tiền nạp tối thiểu là 1,000 VNĐ"}), 400
    elif package_id in packages:
        added_credit = packages[package_id]
    else:
        return jsonify({"error": "Gói nạp không hợp lệ"}), 400
    
    try:
        user_ref = db.collection("user_info").document(uid)
        doc = user_ref.get()
        purchased = 0.0
        if doc.exists:
            data_dict = doc.to_dict()
            purchased = float(data_dict.get("purchased_credit", 0.0))
        
        new_purchased = purchased + added_credit
        user_ref.set({"purchased_credit": new_purchased}, merge=True)
        
        # Save transaction log in database
        db.collection("transactions").add({
            "uid": uid,
            "package_id": package_id,
            "amount": added_credit,
            "payment_method": payment_method,
            "status": "completed",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        
        balances = get_or_init_user_credits(uid)
        print(f"[RECHARGE SUCCESS] Added {added_credit} VND to: {uid}. New balance: {balances['total_credit']} VND")
        return jsonify({
            "success": True, 
            "credits_added": added_credit, 
            "free_credit": balances["free_credit"],
            "purchased_credit": balances["purchased_credit"],
            "total_credit": balances["total_credit"],
            "message": f"Nạp thành công {added_credit:,.0f} VNĐ! Số dư mới: {balances['total_credit']:,.0f} VNĐ."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Admin Middleware ──────────────────────────────────────────────────────────

def require_admin(uid):
    """Returns True if uid belongs to an admin user, raises otherwise."""
    if not uid:
        return False
    try:
        user = fb_auth.get_user(uid)
        return (user.custom_claims or {}).get("role") == "admin"
    except Exception:
        return False

# ── Admin Routes ──────────────────────────────────────────────────────────────

@app.route('/api/admin/users', methods=['POST'])
def admin_list_users():
    uid = request.json.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    try:
        users = []
        page = fb_auth.list_users()
        while page:
            for u in page.users:
                users.append({
                    "uid": u.uid,
                    "email": u.email or "",
                    "role": (u.custom_claims or {}).get("role", "user"),
                    "disabled": u.disabled,
                    "created": u.user_metadata.creation_timestamp if u.user_metadata else None
                })
            page = page.get_next_page()
        return jsonify({"users": users})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/users/role', methods=['POST'])
def admin_set_role():
    data = request.json
    uid = data.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    target_uid = data.get('target_uid')
    new_role = data.get('role')  # 'admin' | 'user'
    if not target_uid or new_role not in ('admin', 'user'):
        return jsonify({"error": "Missing target_uid or invalid role"}), 400
    try:
        existing = fb_auth.get_user(target_uid)
        claims = dict(existing.custom_claims or {})
        claims['role'] = new_role
        fb_auth.set_custom_user_claims(target_uid, claims)
        return jsonify({"success": True, "uid": target_uid, "role": new_role})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/history', methods=['POST'])
def admin_get_history():
    uid = request.json.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    def _serialize(docs):
        logs = []
        for doc in docs:
            d = doc.to_dict()
            d['id'] = doc.id
            ts = d.get('timestamp')
            d['timestamp'] = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None
            logs.append(d)
        return logs

    try:
        docs = db.collection("translation_logs") \
                 .order_by("timestamp", direction=firestore.Query.DESCENDING) \
                 .limit(200).stream()
        logs = _serialize(docs)
        print(f"[ADMIN] Fetched {len(logs)} translation logs (ordered)")
        return jsonify({"history": logs})
    except Exception as e:
        print(f"[ADMIN] order_by failed ({e}), retrying without sort...")
        try:
            docs = db.collection("translation_logs").limit(200).stream()
            logs = _serialize(docs)
            print(f"[ADMIN] Fetched {len(logs)} translation logs (unordered fallback)")
            return jsonify({"history": logs})
        except Exception as e2:
            print(f"[ADMIN] translation_logs fetch failed: {e2}")
            return jsonify({"error": str(e2)}), 500


@app.route('/api/admin/history/delete', methods=['POST'])
def admin_delete_history():
    data = request.json
    uid = data.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    doc_id = data.get('doc_id')
    if not doc_id:
        return jsonify({"error": "Missing doc_id"}), 400
    try:
        db.collection("translation_logs").document(doc_id).delete()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/history/clear', methods=['POST'])
def admin_clear_history():
    uid = request.json.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    try:
        docs = db.collection("translation_logs").stream()
        batch = db.batch()
        count = 0
        for doc in docs:
            batch.delete(doc.reference)
            count += 1
            if count % 500 == 0:
                batch.commit()
                batch = db.batch()
        if count % 500 != 0:
            batch.commit()
        return jsonify({"success": True, "deleted": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/glossary', methods=['POST'])
def admin_get_glossary():
    uid = request.json.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    try:
        docs = db.collection("system_glossary").stream()
        terms = []
        for doc in docs:
            d = doc.to_dict()
            d['id'] = doc.id
            terms.append(d)
        return jsonify({"glossary": terms})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/glossary/add', methods=['POST'])
def admin_add_glossary():
    data = request.json
    uid = data.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    term = data.get('term', '').strip()
    meaning = data.get('meaning', '').strip()
    context = data.get('context', '').strip()
    if not term or not meaning:
        return jsonify({"error": "Missing term or meaning"}), 400
    try:
        entry = {"term": term, "meaning": meaning, "context": context,
                 "timestamp": firestore.SERVER_TIMESTAMP}
        _, ref = db.collection("system_glossary").add(entry)
        return jsonify({"success": True, "id": ref.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/glossary/update', methods=['POST'])
def admin_update_glossary():
    data = request.json
    uid = data.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    doc_id = data.get('id')
    term = data.get('term', '').strip()
    meaning = data.get('meaning', '').strip()
    context = data.get('context', '').strip()
    if not doc_id or not term or not meaning:
        return jsonify({"error": "Missing id, term, or meaning"}), 400
    try:
        db.collection("system_glossary").document(doc_id).update(
            {"term": term, "meaning": meaning, "context": context})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/glossary/delete', methods=['POST'])
def admin_delete_glossary():
    data = request.json
    uid = data.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    doc_id = data.get('id')
    if not doc_id:
        return jsonify({"error": "Missing id"}), 400
    try:
        db.collection("system_glossary").document(doc_id).delete()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Status Route ──────────────────────────────────────────────────────────────

@app.route('/api/status', methods=['GET'])
def status():
    """Check if the API, model, and Firebase are running."""
    models_status = []
    if db is not None:
        try:
            docs = db.collection("AI_model").stream()
            for doc in docs:
                m_data = doc.to_dict()
                mid = m_data.get("model_id")
                name = m_data.get("name", mid)
                if mid:
                    models_status.append({
                        "model_id": mid,
                        "name": name,
                        "loaded": mid in loaded_models,
                        "input_price_1m": float(m_data.get("input_price_1m", 0.0)),
                        "output_price_1m": float(m_data.get("output_price_1m", 0.0))
                    })
        except Exception as e:
            print(f"[Status Error] Failed to fetch models status: {e}")

    return jsonify({
        "status": "running",
        "model_loaded": len(loaded_models) > 0,
        "firebase_connected": db is not None,
        "models": models_status
    })

SEP = " ||| "   # Separator dùng trong batch translation

@app.route('/api/translate/batch', methods=['POST'])
def translate_batch():
    """Batch translation: nhận mảng texts, dịch 1 lần, trả mảng kết quả.
    Gom các đoạn lại bằng SEP, gọi model 1 lần duy nhất, tách kết quả về.
    """
    import time as _t
    _start = _t.perf_counter()

    data = request.json
    if not data or 'texts' not in data:
        return jsonify({"error": "Missing texts array"}), 400

    texts = data['texts']  # list[str]
    if not texts:
        return jsonify({"translations": []}), 200

    target_lang = data.get('target_lang', 'auto')
    user_id     = data.get('user_id', 'anonymous')
    glossary    = data.get('glossary', {})
    glossary_mode = data.get('glossary_mode', 'both')
    model_id    = data.get('model_id', 'qwen3')

    if not user_id or user_id == 'anonymous':
        return jsonify({"error": "Yêu cầu đăng nhập"}), 401

    try:
        model, tokenizer, path = get_model_and_tokenizer(model_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Gom tất cả đoạn thành 1 chuỗi, cách nhau bằng SEP
    combined = SEP.join(t.strip() for t in texts)

    print(f"\n[BATCH TRANSLATE] {len(texts)} segments, {len(combined)} chars total")

    with inference_lock:
        try:
            result = generate_translation(
                model, tokenizer, combined, '',
                target_lang, glossary, glossary_mode, model_id, model_path=path
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Tách kết quả
    parts = [p.strip() for p in result.split(SEP.strip())]

    # Nếu model bỏ sót separator → fallback: trả nguyên chuỗi cho node đầu, blank cho còn lại
    if len(parts) != len(texts):
        print(f"[BATCH TRANSLATE] Separator mismatch: got {len(parts)}, expected {len(texts)} — using raw split fallback")
        # Cố chia đều theo số đoạn
        parts = result.split('|||')
        parts = [p.strip() for p in parts]
        # Nếu vẫn lệch, pad hoặc truncate
        if len(parts) < len(texts):
            parts += [''] * (len(texts) - len(parts))
        parts = parts[:len(texts)]

    elapsed = _t.perf_counter() - _start
    print(f"[BATCH TRANSLATE] Done in {elapsed:.2f}s for {len(texts)} segments")

    return jsonify({"translations": parts, "elapsed": round(elapsed, 2)})


@app.route('/api/translate', methods=['POST'])
def translate():
    """Endpoint to process context-aware translation (auto-detects EN↔VI) with credits and dynamic models."""
    import time as _time_module
    _request_start = _time_module.perf_counter()
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    context = data.get('context', '')
    target_lang = data.get('target_lang', 'auto')  # 'auto' | 'vietnamese' | 'english'
    glossary = data.get('glossary', {})
    glossary_mode = data.get('glossary_mode', 'both')  # 'both' | 'direct' | 'ai'
    user_id = data.get('user_id', 'anonymous')
    model_id = data.get('model_id', 'qwen3') # 'qwen2' | 'qwen3'
    share_translation = data.get('share_translation', False)

    if not user_id or user_id == 'anonymous':
        return jsonify({"error": "Yêu cầu đăng nhập để sử dụng tính năng dịch thuật."}), 401

    is_admin = require_admin(user_id)

    # 1. Dynamic Model Load
    try:
        model, tokenizer, path = get_model_and_tokenizer(model_id)
    except Exception as load_err:
        return jsonify({"error": f"Failed to load model {model_id}: {str(load_err)}"}), 500

    # 2. Dynamic Pricing Setup (VND per token) loaded from Firestore AI_model
    model_pricing = {"input": 5000.0 / 1000000, "output": 15000.0 / 1000000}
    if db is not None:
        try:
            doc = db.collection("AI_model").document(model_id).get()
            if doc.exists:
                d_dict = doc.to_dict()
                model_pricing = {
                    "input": float(d_dict.get("input_price_1m", 5000.0)) / 1000000,
                    "output": float(d_dict.get("output_price_1m", 15000.0)) / 1000000
                }
        except Exception as e:
            print(f"[FIREBASE ERROR] Failed to load dynamic model pricing for {model_id}: {e}")
            
    if model_id == "qwen3" and model_pricing["input"] == 5000.0 / 1000000:
        # Fallback if DB doesn't have qwen3 yet
        model_pricing = {"input": 7000.0 / 1000000, "output": 21000.0 / 1000000}

    # 3. Input token calculation
    if isinstance(tokenizer, str):
        # Online model fallback
        input_tokens = max(1, len(text.split()) * 2)
    else:
        input_tokens = len(tokenizer.encode(text))
    
    # 4. Check Credit Limit
    if not is_admin:
        credits = get_or_init_user_credits(user_id)
        est_input_cost = input_tokens * model_pricing["input"]
        if credits["total_credit"] < est_input_cost:
            return jsonify({
                "error": "OUT_OF_TOKENS",
                "message": f"Bạn đã hết credit dịch thuật (Số dư hiện tại: {credits['total_credit']:,.2f} VNĐ). Vui lòng nạp thêm credit để tiếp tục sử dụng.",
                "free_credit": credits["free_credit"],
                "purchased_credit": credits["purchased_credit"],
                "total_credit": credits["total_credit"]
            }), 402
    else:
        credits = {"free_credit": -1.0, "purchased_credit": -1.0, "total_credit": -1.0}

    # If no glossary sent from client, auto-load from Firestore for logged-in users
    if not glossary:
        glossary = get_user_glossary(user_id)

    # Smart Cache Check - Skip inference if we already have this translation
    if target_lang not in ('explain', 'summarize'):
        cached_translation = check_translation_cache(user_id, text)
        if cached_translation:
            print(f"[TRANSLATE CACHE HIT] Returning cached translation: {repr(cached_translation)}\n")
            return jsonify({
                "translation": cached_translation,
                "target_lang": target_lang,
                "from_cache": True,
                "free_credit": credits["free_credit"],
                "purchased_credit": credits["purchased_credit"],
                "total_credit": credits["total_credit"]
            })

    print(f"\n{'='*60}")
    print(f"[TRANSLATE REQUEST]")
    print(f"  text ({len(text)} chars, {input_tokens} {model_id} input tokens): {repr(text)}")
    print(f"  context ({len(context)} chars): {repr(context)}")
    print(f"  target_lang: {target_lang}")
    if glossary:
        print(f"  glossary ({len(glossary)} items): {list(glossary.keys())} (mode: {glossary_mode})")
    print(f"  model_id: {model_id} (Input Price: {model_pricing['input']*1000000:.0f} VND / 1M, Output: {model_pricing['output']*1000000:.0f} VND / 1M)")
    print(f"  share_translation: {share_translation}")
    print(f"{'='*60}")

    with inference_lock:
        try:
            _t0 = _time_module.perf_counter()
            translation = generate_translation(model, tokenizer, text, context, target_lang, glossary, glossary_mode, model_id, model_path=path)
            _inference_elapsed = _time_module.perf_counter() - _t0
            _total_elapsed = _time_module.perf_counter() - _request_start
            _chars_per_sec = len(translation) / _inference_elapsed if _inference_elapsed > 0 else 0
            print(f"[TRANSLATE RESULT] {repr(translation)}")
            print(f"[TRANSLATE TIMING] total={_total_elapsed:.2f}s | inference={_inference_elapsed:.2f}s | overhead={_total_elapsed - _inference_elapsed:.2f}s | {len(text)} chars in → {len(translation)} chars out | {_chars_per_sec:.0f} chars/s\n")
            
            # Count output tokens
            if isinstance(tokenizer, str):
                output_tokens = max(1, len(translation.split()) * 2)
            else:
                output_tokens = len(tokenizer.encode(translation))
                
            cost_input = input_tokens * model_pricing["input"]
            cost_output = output_tokens * model_pricing["output"]
            total_cost_vnd = cost_input + cost_output
            
            if not is_admin:
                # Deduct credits
                new_balances = deduct_user_credits(user_id, total_cost_vnd)
            else:
                total_cost_vnd = 0.0
                new_balances = {"free_credit": -1.0, "purchased_credit": -1.0, "total_credit": -1.0}
            
            # 1. Always save to user's private history for dashboard access and caching
            threading.Thread(
                target=save_user_private_history,
                args=(user_id, text, translation),
                daemon=True
            ).start()

            # 2. Save log to admin translation_logs only if user agreed to share (anonymously)
            if share_translation:
                threading.Thread(
                    target=save_translation_log,
                    args=(text, translation, target_lang, model_id),
                    daemon=True
                ).start()
            
            return jsonify({
                "translation": translation, 
                "target_lang": target_lang,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_vnd": total_cost_vnd,
                "free_credit": new_balances["free_credit"],
                "purchased_credit": new_balances["purchased_credit"],
                "total_credit": new_balances["total_credit"]
            })
        except Exception as e:
            print(f"[TRANSLATE ERROR] {e}\n")
            return jsonify({"error": str(e)}), 500


# Lazy loader for EasyOCR to prevent memory overhead at startup
easyocr_reader = None

def get_easyocr_reader():
    global easyocr_reader
    if easyocr_reader is None:
        import easyocr
        print("[EasyOCR] Initializing EasyOCR Reader (vi, en)...")
        # Automatically detects CUDA/GPU if available, otherwise runs on CPU
        easyocr_reader = easyocr.Reader(['vi', 'en'])
        print("[EasyOCR] EasyOCR Reader initialized successfully!")
    return easyocr_reader


def preload_easyocr_in_background():
    try:
        import time
        time.sleep(5)
        print("[EasyOCR] Background preloader starting...")
        get_easyocr_reader()
    except Exception as e:
        print(f"[EasyOCR Preload Error] Failed background preload: {e}")

# Start preloading EasyOCR vi/en models in a background thread at startup
threading.Thread(target=preload_easyocr_in_background, daemon=True).start()


@app.route('/api/ocr', methods=['POST'])
def ocr_image():
    """OCR an image sent as base64 and return extracted text using EasyOCR."""
    import base64 as _b64, io
    from PIL import Image
    import numpy as np

    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image data"}), 400

    image_b64 = data['image']
    if ',' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]

    try:
        img_bytes = _b64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        img_np = np.array(img)

        # Lazy load the EasyOCR reader
        reader = get_easyocr_reader()

        # Run OCR
        results = reader.readtext(img_np, detail=0)
        text = " ".join(results).strip()

        if not text:
            return jsonify({"error": "Không tìm thấy chữ trong ảnh"}), 422

        print(f"[EasyOCR] Extracted {len(text)} chars from image")
        return jsonify({"text": text})
    except Exception as e:
        print(f"[EasyOCR ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/document/extract', methods=['POST'])
def extract_document():
    """Extract text from uploaded PDF or DOCX file."""
    import io
    from pypdf import PdfReader
    from docx import Document

    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy tài liệu"}), 400

    file = request.files['file']
    filename = file.filename.lower()

    try:
        file_stream = io.BytesIO(file.read())
        text = ""

        if filename.endswith('.pdf'):
            reader = PdfReader(file_stream)
            text_list = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_list.append(extracted)
            text = "\n".join(text_list)
        elif filename.endswith('.docx'):
            doc = Document(file_stream)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            return jsonify({"error": "Định dạng file không hỗ trợ. Chỉ nhận PDF và DOCX."}), 400

        text = text.strip()
        if not text:
            return jsonify({"error": "Không thể trích xuất chữ hoặc tài liệu rỗng"}), 422

        print(f"[DOC EXTRACT] Extracted {len(text)} chars from {file.filename}")
        return jsonify({"text": text})
    except Exception as e:
        print(f"[DOC EXTRACT ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def detect():
    """Lightweight endpoint to detect language without running inference."""
    from model_inference import detect_language
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    lang = detect_language(data['text'])
    return jsonify({"language": lang})

# ── Translation Rating & Evaluation Routes ────────────────────────────────────

@app.route('/api/translate/rate', methods=['POST'])
def rate_translation():
    """Rate a translation (like/dislike) in user's history."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    data = request.json
    uid = data.get('uid')
    source = data.get('source')
    rating = data.get('rating')  # 'like' | 'dislike' | None
    
    if not uid or not source:
        return jsonify({"error": "Missing uid or source"}), 400
        
    try:
        # Fetch matching documents
        docs = db.collection("users").document(uid).collection("history")\
                 .where("source", "==", source.strip()).stream()
        
        found = False
        for doc in docs:
            found = True
            if rating == 'dislike':
                doc.reference.delete()
            else:
                doc.reference.update({"rating": rating})
            
        return jsonify({"success": True, "found": found})
    except Exception as e:
        print(f"[RATE TRANSLATION ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# ── Saved Translations Routes ──────────────────────────────────────────────────

@app.route('/api/saved_translations', methods=['POST'])
def get_saved_translations():
    """Fetch user's saved translations from Firestore."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    uid = request.json.get('uid')
    if not uid:
        return jsonify({"error": "Missing uid"}), 400
    try:
        docs = db.collection("users").document(uid).collection("saved_translations")\
                 .order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        saved_list = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            ts = data.get('timestamp')
            data['timestamp'] = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None
            saved_list.append(data)
        return jsonify({"saved_translations": saved_list})
    except Exception as e:
        # Fallback without sort if index not ready
        try:
            docs = db.collection("users").document(uid).collection("saved_translations").stream()
            saved_list = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                ts = data.get('timestamp')
                data['timestamp'] = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None
                saved_list.append(data)
            return jsonify({"saved_translations": saved_list})
        except Exception as e2:
            return jsonify({"error": str(e2)}), 500

@app.route('/api/saved_translations/add', methods=['POST'])
def add_saved_translation():
    """Save a translation with a note."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    data = request.json
    uid = data.get('uid')
    source_text = data.get('source_text')
    translated_text = data.get('translated_text')
    note = data.get('note', '')

    if not uid or not source_text or not translated_text:
        return jsonify({"error": "Missing required fields"}), 400
    
    new_entry = {
        "source_text": source_text.strip(),
        "translated_text": translated_text.strip(),
        "note": note,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    try:
        _, doc_ref = db.collection("users").document(uid).collection("saved_translations").add(new_entry)
        new_entry['id'] = doc_ref.id
        new_entry['timestamp'] = None
        return jsonify({"success": True, "entry": new_entry})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/saved_translations/update_note', methods=['POST'])
def update_saved_translation_note():
    """Update note of a saved translation."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    data = request.json
    uid = data.get('uid')
    doc_id = data.get('id')
    note = data.get('note', '')

    if not uid or not doc_id:
        return jsonify({"error": "Missing uid or id"}), 400
    try:
        db.collection("users").document(uid).collection("saved_translations").document(doc_id).update({"note": note})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/saved_translations/delete', methods=['POST'])
def delete_saved_translation():
    """Delete a saved translation."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    data = request.json
    uid = data.get('uid')
    doc_id = data.get('id')

    if not uid or not doc_id:
        return jsonify({"error": "Missing uid or id"}), 400
    try:
        db.collection("users").document(uid).collection("saved_translations").document(doc_id).delete()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Contributions Routes ────────────────────────────────────────────────────────

@app.route('/api/translate/contribute', methods=['POST'])
def contribute_translation():
    """Contribute a translation to the global contributions database."""
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    data = request.json
    uid = data.get('uid', 'anonymous')
    email = data.get('email', 'anonymous')
    source_text = data.get('source_text')
    original_translation = data.get('original_translation')
    suggested_translation = data.get('suggested_translation')

    if not source_text or not suggested_translation:
        return jsonify({"error": "Missing required fields"}), 400

    new_contribution = {
        "user_id": uid,
        "email": email,
        "source_text": source_text.strip(),
        "original_translation": original_translation.strip() if original_translation else "",
        "suggested_translation": suggested_translation.strip(),
        "timestamp": firestore.SERVER_TIMESTAMP,
        "status": "pending"  # 'pending' | 'approved' | 'rejected'
    }
    try:
        _, doc_ref = db.collection("contributions").add(new_contribution)
        
        # Mark history entry as contributed
        if uid and uid != 'anonymous':
            try:
                docs = db.collection("users").document(uid).collection("history")\
                         .where("source", "==", source_text.strip()).limit(5).stream()
                for doc in docs:
                    doc.reference.update({"is_contributed": True})
            except Exception as hist_err:
                print(f"[CONTRIBUTE] Failed to mark history entry: {hist_err}")

        return jsonify({"success": True, "id": doc_ref.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/contributions', methods=['POST'])
def admin_get_contributions():
    uid = request.json.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    try:
        docs = db.collection("contributions")\
                 .order_by("timestamp", direction=firestore.Query.DESCENDING).limit(100).stream()
        contributions = []
        for doc in docs:
            d = doc.to_dict()
            d['id'] = doc.id
            ts = d.get('timestamp')
            d['timestamp'] = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None
            contributions.append(d)
        return jsonify({"contributions": contributions})
    except Exception as e:
        # Fallback without sorting
        try:
            docs = db.collection("contributions").limit(100).stream()
            contributions = []
            for doc in docs:
                d = doc.to_dict()
                d['id'] = doc.id
                ts = d.get('timestamp')
                d['timestamp'] = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None
                contributions.append(d)
            return jsonify({"contributions": contributions})
        except Exception as e2:
            return jsonify({"error": str(e2)}), 500

@app.route('/api/admin/contributions/action', methods=['POST'])
def admin_contribution_action():
    data = request.json
    uid = data.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    
    contrib_id = data.get('id')
    action = data.get('action')  # 'approved' | 'rejected' | 'delete'
    if not contrib_id or not action:
        return jsonify({"error": "Missing id or action"}), 400
        
    try:
        ref = db.collection("contributions").document(contrib_id)
        if action == 'delete':
            ref.delete()
        else:
            ref.update({"status": action})
            
            # If approved, add it to system glossary
            if action == 'approved':
                contrib_doc = ref.get()
                if contrib_doc.exists:
                    cdata = contrib_doc.to_dict()
                    term = cdata.get("source_text", "").strip()
                    meaning = cdata.get("suggested_translation", "").strip()
                    if term and meaning:
                        db.collection("system_glossary").add({
                            "term": term,
                            "meaning": meaning,
                            "context": f"Đóng góp từ {cdata.get('email', 'User')}",
                            "timestamp": firestore.SERVER_TIMESTAMP
                        })
                        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models', methods=['GET'])
def list_models_public():
    """List all available models dynamically for standard users dropdown."""
    if db is None:
        return jsonify({"models": [
            {"model_id": "qwen2", "name": "Qwen2-1.5b", "input_price_1m": 5000.0, "output_price_1m": 15000.0},
            {"model_id": "qwen3", "name": "Qwen3-1.7b", "input_price_1m": 7000.0, "output_price_1m": 21000.0}
        ]})
    try:
        docs = db.collection("AI_model").stream()
        models = []
        for doc in docs:
            d = doc.to_dict()
            models.append({
                "model_id": d.get("model_id"),
                "name": d.get("name"),
                "input_price_1m": d.get("input_price_1m"),
                "output_price_1m": d.get("output_price_1m")
            })
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/models', methods=['POST'])
def admin_list_models():
    uid = request.json.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    try:
        docs = db.collection("AI_model").stream()
        models = []
        for doc in docs:
            d = doc.to_dict()
            d['id'] = doc.id
            models.append(d)
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/transactions', methods=['POST'])
def admin_list_transactions():
    uid = request.json.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    try:
        docs = db.collection("transactions").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(100).stream()
        transactions = []
        for doc in docs:
            d = doc.to_dict()
            if d.get("status") == "pending":
                continue
            d['id'] = doc.id
            ts = d.get('timestamp')
            d['timestamp'] = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None
            transactions.append(d)
        return jsonify({"transactions": transactions})
    except Exception as e:
        try:
            docs = db.collection("transactions").limit(100).stream()
            transactions = []
            for doc in docs:
                d = doc.to_dict()
                if d.get("status") == "pending":
                    continue
                d['id'] = doc.id
                ts = d.get('timestamp')
                d['timestamp'] = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None
                transactions.append(d)
            return jsonify({"transactions": transactions})
        except Exception as e2:
            return jsonify({"error": str(e2)}), 500

@app.route('/api/admin/models/add', methods=['POST'])
def admin_add_model():
    data = request.json
    uid = data.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    
    model_id = data.get('model_id', '').strip()
    name = data.get('name', '').strip()
    path = data.get('path', '').strip()
    input_price = float(data.get('input_price_1m', 5000.0))
    output_price = float(data.get('output_price_1m', 15000.0))
    
    if not model_id or not name or not path:
        return jsonify({"error": "Missing required fields (model_id, name, path)"}), 400
        
    try:
        db.collection("AI_model").document(model_id).set({
            "model_id": model_id,
            "name": name,
            "path": path,
            "input_price_1m": input_price,
            "output_price_1m": output_price
        })
        return jsonify({"success": True, "model_id": model_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/models/delete', methods=['POST'])
def admin_delete_model():
    data = request.json
    uid = data.get('uid')
    if not require_admin(uid):
        return jsonify({"error": "Forbidden"}), 403
    if db is None:
        return jsonify({"error": "Firebase not connected"}), 500
    
    model_id = data.get('model_id')
    if not model_id:
        return jsonify({"error": "Missing model_id"}), 400
        
    try:
        db.collection("AI_model").document(model_id).delete()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/payment/create', methods=['POST'])
def create_payment():
    if db is None or payos_client is None:
        return jsonify({"error": "Payment service not configured or Firebase offline"}), 500
    
    data = request.json
    uid = data.get('uid')
    amount = data.get('amount')
    package_id = data.get('package_id')
    
    if not uid or not amount:
        return jsonify({"error": "Missing uid or amount"}), 400
        
    try:
        amount = int(amount)
    except ValueError:
        return jsonify({"error": "Invalid amount"}), 400

    import time
    # Order code must be a number and max 9007199254740991
    order_code = int(time.time() * 1000) % 9007199254740991

    # Default redirect URLs pointing to the dashboard
    return_url = data.get('return_url', "https://hvmndoan-production.up.railway.app/src/dashboard/index.html")
    cancel_url = data.get('cancel_url', "https://hvmndoan-production.up.railway.app/src/dashboard/index.html")
    
    # Store pending transaction info in Firestore
    db.collection("transactions").document(str(order_code)).set({
        "uid": uid,
        "amount": amount,
        "package_id": package_id,
        "status": "pending",
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    
    # Description limit is 25 characters
    desc = f"Nap {package_id} {uid[:8]}".strip()
    if len(desc) > 25:
        desc = desc[:25]
        
    try:
        # Create payment data using official PayOS Types
        item = ItemData(name=f"Gói {package_id}", quantity=1, price=amount)
        payment_data = CreatePaymentLinkRequest(
            order_code=order_code,
            amount=amount,
            description=desc,
            items=[item],
            return_url=return_url,
            cancel_url=cancel_url
        )
        
        # Try both direct method and payment_requests namespace
        try:
            res = payos_client.create_payment_link(payment_data)
        except AttributeError:
            res = payos_client.payment_requests.create(payment_data)
            
        return jsonify({
            "success": True,
            "checkoutUrl": getattr(res, "checkout_url", getattr(res, "checkoutUrl", None)),
            "orderCode": order_code
        })
    except Exception as e:
        print(f"[PayOS Error] {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/payment/webhook', methods=['POST'])
def payos_webhook():
    print("[Webhook Debug] Received webhook request!")
    if db is None or payos_client is None:
        print("[Webhook Error] Firebase database or PayOS client is None!")
        return jsonify({"error": "Payment service not configured or Firebase offline"}), 500
        
    webhook_data = request.json
    print(f"[Webhook Debug] Request JSON data: {webhook_data}")
    try:
        verified_data = None
        order_code = None
        amount = None
        
        # Robust verification supporting both verify_payment_webhook_data and webhooks.verify
        try:
            print("[Webhook Debug] Attempting verification via verify_payment_webhook_data...")
            verified_data = payos_client.verify_payment_webhook_data(webhook_data)
            print(f"[Webhook Debug] verify_payment_webhook_data success. Data: {verified_data}")
            order_code = str(verified_data.get('orderCode') or verified_data.get('order_code'))
            amount = verified_data.get('amount')
        except Exception as e1:
            print(f"[Webhook Warning] verify_payment_webhook_data failed: {e1}. Retrying with webhooks.verify...")
            try:
                raw_data = request.get_data()
                print(f"[Webhook Debug] Raw request bytes: {raw_data}")
                verified_data = payos_client.webhooks.verify(raw_data)
                print(f"[Webhook Debug] webhooks.verify success. Data: {verified_data}")
                order_code = str(verified_data.order_code)
                amount = verified_data.amount
            except Exception as e2:
                print(f"[Webhook Error] Both verification methods failed! verify_payment_webhook_data err: {e1}, webhooks.verify err: {e2}")
                raise Exception(f"Signature verification failed. e1: {e1}, e2: {e2}")
            
        print(f"[Webhook Debug] Extracted order_code: {order_code}, amount: {amount}")
        
        # Get transaction details from Firestore
        order_ref = db.collection("transactions").document(order_code)
        order_snap = order_ref.get()
        print(f"[Webhook Debug] Firestore check for transaction {order_code}: exists={order_snap.exists}")
        if not order_snap.exists:
            print(f"[Webhook Info] Transaction {order_code} not found in collection 'transactions'. This might be a test/dummy webhook from PayOS registration. Returning 200.")
            return jsonify({"success": True, "message": "Transaction not found but signature verified (possibly test/dummy webhook)"}), 200
            
        order_info = order_snap.to_dict()
        print(f"[Webhook Debug] Transaction info from DB: {order_info}")
        if order_info.get('status') == 'completed':
            print(f"[Webhook Info] Transaction {order_code} status is already 'completed'. Skipping credit addition.")
            return jsonify({"success": True, "message": "Already processed"}), 200
            
        uid = order_info.get('uid')
        package_id = order_info.get('package_id')
        print(f"[Webhook Debug] Transaction belongs to user (uid): {uid}, package_id: {package_id}")
        if not uid:
            print("[Webhook Error] No uid found in transaction info!")
            return jsonify({"error": "No uid in transaction info"}), 400
            
        # 1 VNĐ = 1 Credit
        credits_to_add = amount
        print(f"[Webhook Debug] Credits to add: {credits_to_add} (from amount: {amount})")
        
        user_ref = db.collection("user_info").document(uid)
        
        # Use a transaction to safely update credits
        @firestore.transactional
        def update_credits_tx(transaction, user_ref, credits_to_add, amount, package_id):
            print(f"[Webhook Tx] Reading user profile for {uid} inside transaction...")
            user_snap = user_ref.get(transaction=transaction)
            if not user_snap.exists:
                print(f"[Webhook Tx] User {uid} does not exist in 'user_info' collection. Creating with default + purchased credits.")
                transaction.set(user_ref, {
                    "free_credit": 100000.0,
                    "purchased_credit": float(credits_to_add),
                    "total_credit": 100000.0 + float(credits_to_add),
                    "role": "user",
                    "email": ""
                })
            else:
                user_data = user_snap.to_dict()
                current_purchased = user_data.get("purchased_credit", 0.0)
                current_free = user_data.get("free_credit", 100000.0)
                print(f"[Webhook Tx] Current user credit state: free={current_free}, purchased={current_purchased}")
                
                new_purchased = current_purchased + float(credits_to_add)
                new_total = current_free + new_purchased
                print(f"[Webhook Tx] Updating user {uid} credit state to: purchased={new_purchased}, total={new_total}")
                transaction.update(user_ref, {
                    "purchased_credit": new_purchased,
                    "total_credit": new_total
                })
            
        transaction = db.transaction()
        print("[Webhook Debug] Starting Firestore transaction...")
        update_credits_tx(transaction, user_ref, credits_to_add, amount, package_id)
        print("[Webhook Debug] Firestore transaction completed successfully!")
        
        # Update transaction status to completed
        print(f"[Webhook Debug] Updating transaction {order_code} status to 'completed'...")
        order_ref.update({
            "status": "completed",
            "payment_method": "payos",
            "completed_at": firestore.SERVER_TIMESTAMP
        })
        print(f"[Webhook Debug] Transaction {order_code} status successfully updated to completed.")
        
        return jsonify({"success": True, "message": "Payment verified and credited"}), 200
    except Exception as e:
        import traceback
        print(f"[Webhook Error] Exception occurred:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    print("Pre-loading Qwen2 model...")
    try:
        get_model_and_tokenizer('qwen2')
    except Exception as e:
        print(f"Warning: Failed to pre-load Qwen2 at startup: {e}")
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)

