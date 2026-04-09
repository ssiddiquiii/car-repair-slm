import os
from dotenv import load_dotenv
from huggingface_hub import login

def setup_environment():
    print("Initializing Environment...")
    load_dotenv() # Loads keys from your root .env file

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not GEMINI_API_KEY or not HF_TOKEN:
        raise ValueError("Missing API Keys in .env file!")

    login(token=HF_TOKEN)
    print("Keys loaded & HuggingFace authenticated ✅")
    
    return GEMINI_API_KEY