"""
Phase I — Download Llama-3.2-1B-Instruct from Hugging Face to a local directory.

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login   (paste your HF token when prompted)

Run:
  python download_model.py
"""

from huggingface_hub import snapshot_download
import os

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LOCAL_DIR = "./models/llama-3.2-1b-instruct"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Downloading {MODEL_ID} to {LOCAL_DIR} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        ignore_patterns=["*.pt", "original/*"],  # skip full-precision weights
    )
    print(f"\nDone! Model saved to: {os.path.abspath(LOCAL_DIR)}")

if __name__ == "__main__":
    main()
