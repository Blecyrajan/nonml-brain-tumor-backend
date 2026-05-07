import os
import requests
import google.auth
import google.auth.transport.requests
from dotenv import load_dotenv

# ======================================================
# LOAD ENV VARIABLES
# ======================================================

load_dotenv()

# ======================================================
# CREATE GCP KEY FILE FROM RENDER ENV VARIABLE
# ======================================================

gcp_creds = os.getenv("GCP_CREDENTIALS")

if gcp_creds:

    with open("gcp_key.json", "w") as f:
        f.write(gcp_creds)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

# ======================================================
# VERTEX AI ENDPOINT
# ======================================================

VERTEX_ENDPOINT = os.getenv("VERTEX_ENDPOINT")

# ======================================================
# SYSTEM PROMPT
# ======================================================

SYSTEM_PROMPT = (
    "You are a medical AI assistant. "
    "Only provide educational explanations. "
    "Do not provide diagnosis, treatment, or medical advice. "
    "Explain concepts in simple language suitable for patients. "
    "Do not stop mid-sentence."
)

# ======================================================
# GET ACCESS TOKEN
# ======================================================

def get_access_token():

    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform"
        ]
    )

    auth_req = google.auth.transport.requests.Request()

    credentials.refresh(auth_req)

    return credentials.token

# ======================================================
# ASK LLM
# ======================================================

def ask_biomistral(question: str) -> str:

    try:

        # ==========================================
        # GET GOOGLE ACCESS TOKEN
        # ==========================================

        access_token = get_access_token()

        # ==========================================
        # REQUEST HEADERS
        # ==========================================

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        # ==========================================
        # PAYLOAD
        # ==========================================

        payload = {

            "instances": [
                {
                    "inputs": f"""
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

        # ==========================================
        # CALL VERTEX AI ENDPOINT
        # ==========================================

        response = requests.post(
            VERTEX_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=120
        )

        # ==========================================
        # ERROR CHECK
        # ==========================================

        if response.status_code != 200:

            print(
                "VERTEX ERROR:",
                response.status_code,
                response.text
            )

            return (
                "The medical assistant is currently unavailable."
            )

        # ==========================================
        # PARSE RESPONSE
        # ==========================================

        data = response.json()

        print("VERTEX RESPONSE:", data)

        # ==========================================
        # HANDLE PREDICTIONS
        # ==========================================

        if "predictions" in data:

            pred = data["predictions"][0]

            # --------------------------------------
            # STRING RESPONSE
            # --------------------------------------

            if isinstance(pred, str):
                return pred

            # --------------------------------------
            # DICTIONARY RESPONSE
            # --------------------------------------

            if isinstance(pred, dict):

                if "content" in pred:
                    return pred["content"]

                if "generated_text" in pred:
                    return pred["generated_text"]

                if "outputs" in pred:
                    return pred["outputs"]

                if "prediction" in pred:
                    return pred["prediction"]

                return str(pred)

            return str(pred)

        # ==========================================
        # FALLBACK
        # ==========================================

        return "Unable to generate a response."

    # ==================================================
    # EXCEPTION HANDLING
    # ==================================================

    except Exception as e:

        print("VERTEX REQUEST FAILED:", e)

        return (
            "Medical assistant temporarily unavailable."
        )