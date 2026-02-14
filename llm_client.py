import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://router.huggingface.co/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = (
    "You are a medical AI assistant. "
    "Only provide educational explanations. "
    "Do not provide diagnosis, treatment, or medical advice. "
    "Explain concepts in simple language suitable for patients."
    "Do not stop mid-sentence."
)

def ask_biomistral(question: str) -> str:
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2:together",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.4,
        "max_tokens": 1000
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        print("HF ERROR:", response.status_code, response.text)
        return "The medical assistant is currently unavailable."

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print("HF PARSE ERROR:", data)
        return "Unable to generate a response."
