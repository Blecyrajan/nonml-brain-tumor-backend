import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

# Direct endpoint for MedGemma
API_URL = "https://api-inference.huggingface.co/models/google/medgemma-1.5-4b-it"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# This prompt guides the LLM to behave correctly during your Viva demo
SYSTEM_PROMPT = (
    "You are a medical AI assistant. Provide educational explanations about brain tumors "
    "and neurology. Do not provide diagnosis or treatment advice. "
    "Use simple, layman terms. Be concise and do not stop mid-sentence."
)

def ask_med_assistant(question: str) -> str:
    # Formatting for MedGemma/Gemma instruction tuning
    prompt = f"<bos><start_of_turn>user\n{SYSTEM_PROMPT}\n\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.4,
            "top_p": 0.9,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True  # Crucial: Tells HF to load the model if it's idle
        }
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=90)
        
        if response.status_code == 200:
            data = response.json()
            # The serverless API returns a list of dicts
            return data[0]['generated_text'].strip()
        
        elif response.status_code == 503:
            return "The model is currently loading on the server. Please try again in a minute."
        else:
            print(f"HF Error {response.status_code}: {response.text}")
            return "I'm having trouble connecting to my medical database right now."

    except Exception as e:
        print(f"Connection Error: {e}")
        return "System busy. Please try again later."