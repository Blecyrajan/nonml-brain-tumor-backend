import os
import requests
import google.auth
import google.auth.transport.requests

from dotenv import load_dotenv

load_dotenv()

# ======================================================
# VERTEX AI ENDPOINT
# ======================================================

VERTEX_ENDPOINT = os.getenv("VERTEX_ENDPOINT")

SYSTEM_PROMPT = (
    "You are a medical AI assistant. "
    "Only provide educational explanations. "
    "Do not provide diagnosis, treatment, or medical advice. "
    "Explain concepts in simple language suitable for patients."
)

# ======================================================
# GET ACCESS TOKEN
# ======================================================

def get_access_token():

    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    auth_req = google.auth.transport.requests.Request()

    credentials.refresh(auth_req)

    return credentials.token


# ======================================================
# ASK MODEL
# ======================================================

def ask_biomistral(question: str) -> str:

    try:

        access_token = get_access_token()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "instances": [
                {
                    "prompt": f"""
                    System:
                    {SYSTEM_PROMPT}

                    User:
                    {question}
                    """
                }
            ],

            "parameters": {
                "temperature": 0.4,
                "maxOutputTokens": 1000
            }
        }

        response = requests.post(
            VERTEX_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:

            print(
                "VERTEX ERROR:",
                response.status_code,
                response.text
            )

            return "Medical assistant unavailable."

        data = response.json()

        print("VERTEX RESPONSE:", data)

        # ==========================================
        # MODIFY BASED ON MODEL RESPONSE
        # ==========================================

        if "predictions" in data:

            pred = data["predictions"][0]

            if isinstance(pred, dict):

                if "content" in pred:
                    return pred["content"]

                elif "generated_text" in pred:
                    return pred["generated_text"]

            return str(pred)

        return "Unable to generate response."

    except Exception as e:

        print("VERTEX REQUEST FAILED:", e)

        return "Medical assistant temporarily unavailable."