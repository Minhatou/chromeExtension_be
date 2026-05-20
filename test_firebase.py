import firebase_admin
from firebase_admin import credentials, firestore

try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print(">>> FIREBASE CONNECTION OK <<<")

    # Try reading first to detect if Firestore is accessible at all
    print("Trying to READ from Firestore...")
    docs = list(db.collection("test_connection").stream())
    print(f"Read OK - found {len(docs)} documents in 'test_connection'")

    # Now try writing
    print("Trying to WRITE to Firestore...")
    test_ref = db.collection("test_connection").document("status")
    test_ref.set({
        "message": "Hello from Flask Admin SDK!",
        "status": "connected",
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    print(">>> WRITE SUCCESSFUL! Connection fully working! <<<")

except Exception as e:
    import traceback
    print(f"\n[ERROR] {str(e)}")
    print("\n--- Full traceback ---")
    traceback.print_exc()
    print("---")
    print("\nPossible fixes:")
    print("1. Make sure Firestore Database has been CREATED in Firebase Console")
    print("2. Go to Firebase Console -> Firestore Database -> Rules -> set 'allow read, write: if true' -> Publish")
    print("3. Wait 1-2 minutes after enabling Firestore API and try again")
