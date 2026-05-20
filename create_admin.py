"""
Script to create an admin account in Firebase Authentication.
Run once: python create_admin.py
"""
import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# ── CONFIG: Change these to your desired admin credentials ──────────────────
ADMIN_EMAIL    = "nhatcreeper511@gmail.com"  # <-- Change to your email
ADMIN_PASSWORD = "123456"             # <-- Change to your password
ADMIN_NAME     = "Admin"
# ─────────────────────────────────────────────────────────────────────────────

def create_admin_user(email, password, display_name):
    try:
        # Try to create new user
        user = auth.create_user(
            email=email,
            password=password,
            display_name=display_name,
            email_verified=True
        )
        print(f"[OK] Account created: {user.uid}")
    except auth.EmailAlreadyExistsError:
        # If already exists, just get the existing user
        user = auth.get_user_by_email(email)
        print(f"[OK] Account already exists: {user.uid}")

    # Set admin custom claim
    auth.set_custom_user_claims(user.uid, {"role": "admin"})
    print(f"[OK] Admin role granted to: {email}")
    print(f"\n--- ACCOUNT INFO ---")
    print(f"  UID   : {user.uid}")
    print(f"  Email : {email}")
    print(f"  Role  : admin")
    print(f"\nDone! You can now log in with these credentials in the Chrome Extension.")

create_admin_user(ADMIN_EMAIL, ADMIN_PASSWORD, ADMIN_NAME)
