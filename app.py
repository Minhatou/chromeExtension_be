from flask import Flask, request, jsonify
from flask_cors import CORS
from model_inference import load_model, generate_translation
import threading

# ── Firebase Admin SDK ────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore, auth as fb_auth

try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("[FIREBASE] Connected to Cloud Firestore!")
except Exception as e:
    print(f"[FIREBASE ERROR] {e}")
    db = None
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        res = app.make_response('')
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        res.status_code = 200
        return res

# Global variables to hold model and tokenizer
model = None
tokenizer = None

# Thread lock to prevent concurrent VRAM usage and OOM errors during inference
inference_lock = threading.Lock()


# ── Firebase Helper Functions ─────────────────────────────────────────────────

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


def save_translation_log(user_id, source, result, task_type):
    """Save translation log to Firestore asynchronously."""
    if db is None:
        return
    try:
        db.collection("translation_logs").add({
            "user_id": user_id,
            "source_text": source,
            "translated_text": result,
            "type": task_type,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"  [FIREBASE ERROR] Failed to save log: {e}")

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
        
        # Get custom claims (role) using Admin SDK
        user = fb_auth.get_user(uid)
        role = user.custom_claims.get("role", "user") if user.custom_claims else "user"
        
        return jsonify({
            "uid": uid,
            "email": email,
            "role": role,
            "idToken": id_token
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
        
        return jsonify({
            "uid": uid,
            "email": email,
            "role": "user",
            "idToken": id_token
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
        print(f"[AUTH] Token verified: uid={uid} email={email} role={role}")
        return jsonify({"valid": True, "uid": uid, "email": email, "role": role})
    except fb_auth.ExpiredIdTokenError:
        return jsonify({"valid": False, "error": "Token expired"}), 401
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 401


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
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "firebase_connected": db is not None
    })

@app.route('/api/translate', methods=['POST'])
def translate():
    """Endpoint to process context-aware translation (auto-detects EN↔VI)."""
    global model, tokenizer

    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    context = data.get('context', '')
    target_lang = data.get('target_lang', 'auto')  # 'auto' | 'vietnamese' | 'english'
    glossary = data.get('glossary', {})
    glossary_mode = data.get('glossary_mode', 'both')  # 'both' | 'direct' | 'ai'
    user_id = data.get('user_id', 'anonymous')

    # If no glossary sent from client, auto-load from Firestore for logged-in users
    if not glossary and user_id != 'anonymous':
        glossary = get_user_glossary(user_id)

    print(f"\n{'='*60}")
    print(f"[TRANSLATE REQUEST]")
    print(f"  text ({len(text)} chars): {repr(text)}")
    print(f"  context ({len(context)} chars): {repr(context)}")
    print(f"  target_lang: {target_lang}")
    if glossary:
        print(f"  glossary ({len(glossary)} items): {list(glossary.keys())} (mode: {glossary_mode})")
    print(f"{'='*60}")

    with inference_lock:
        try:
            translation = generate_translation(model, tokenizer, text, context, target_lang, glossary, glossary_mode)
            print(f"[TRANSLATE RESULT] {repr(translation)}\n")
            # Save log to Firestore in background thread (non-blocking)
            if user_id != 'anonymous':
                threading.Thread(
                    target=save_translation_log,
                    args=(user_id, text, translation, target_lang),
                    daemon=True
                ).start()
            return jsonify({"translation": translation, "target_lang": target_lang})
        except Exception as e:
            print(f"[TRANSLATE ERROR] {e}\n")
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

if __name__ == '__main__':
    # ── Model config ──────────────────────────────────────────────────────
    BASE_MODEL_PATH   = r"C:\Users\Cko Ckeems Ngoo\LlamaFactory\saves\train_qwen_2000cau"
    # Set to adapter path after training, or None to use the base model only:
    LORA_WEIGHTS_PATH = None
    # ─────────────────────────────────────────────────────────────────────

    print("Loading model, please wait...")
    model, tokenizer = load_model(
        base_model_name=BASE_MODEL_PATH,
        lora_weights_path=LORA_WEIGHTS_PATH,
    )
    print("Model loaded. Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)

