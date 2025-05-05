# src/helper.py

import re
import os
import json
from dotenv import load_dotenv

# ========== 1. Cleaning text function ==========

def clean_text(text):
    """Clean unwanted patterns from raw extracted text."""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'(Page \d+|Figure \d+|Table \d+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?:\/\/\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Remove numeric-only lines
    text = re.sub(r'(Copyright|Elsevier|Permissions|ISBN|Editor.*?Edition)', '', text, flags=re.IGNORECASE)
    return text.strip()

# ========== 2. API loading ==========

def load_env_variables():
    """Load .env file and return important API keys."""
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return pinecone_api_key, openai_api_key

# ========== 3. Save and Load JSON file ==========

def save_json(filepath, data):
    """Save dictionary or list to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    """Load data from a JSON file, return empty list if not found."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

# ========== 4. Sanitize filenames for saving ==========

def sanitize_filename(name):
    """Remove problematic characters from filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)
