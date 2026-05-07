import os
import json
from dotenv import load_dotenv

from google.cloud import aiplatform

# ======================================================
# LOAD ENV
# ======================================================

load_dotenv()

# ======================================================
# CREATE GCP KEY FILE
# ======================================================

gcp_creds = os.getenv("GCP_CREDENTIALS")

if not gcp_creds:
    raise Exception("GCP_CREDENTIALS missing")

# validate json
creds_dict = json.loads(gcp_creds)

with open("gcp_key.json", "w") as f:
    json.dump(creds_dict, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

# ======================================================
# CONFIG
# ======================================================

PROJECT_ID = os.getenv("GCP_PROJECT_ID")

LOCATION = os.getenv("GCP_LOCATION")

ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID")

# ======================================================
# INIT VERTEX AI
# ======================================================

aiplatform.init(
    project=PROJECT_ID,
    location=LOCATION
)

# ======================================================
# LOAD ENDPOINT
# ======================================================

endpoint = aiplatform.Endpoint(
    endpoint_name=
    f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
)

# ======================================================
# SYSTEM PROMPT
# ======================================================

SYSTEM_PROMPT = """
You are a medical AI assistant.

Only provide educational explanations.

Do not provide diagnosis,
treatment,
or medical advice.

Explain concepts in simple language.
"""

# ======================================================
# ASK MODEL
# ======================================================

def ask_biomistral(question: str):

    try:

        instances = [
            {
                "prompt": f"""
                {SYSTEM_PROMPT}

                User: {question}

                Assistant:
                """
            }
        ]

        response = endpoint.predict(
            instances=instances
        )

        print("VERTEX RESPONSE:", response)

        predictions = response.predictions

        if len(predictions) == 0:
            return "No response generated."

        pred = predictions[0]

        print("PRED:", pred)

        # ======================================
        # STRING RESPONSE
        # ======================================

        if isinstance(pred, str):
            return pred

        # ======================================
        # DICT RESPONSE
        # ======================================

        if isinstance(pred, dict):

            if "generated_text" in pred:
                return pred["generated_text"]

            if "content" in pred:
                return pred["content"]

            if "output" in pred:
                return pred["output"]

            if "response" in pred:
                return pred["response"]

            return str(pred)

        return str(pred)

    except Exception as e:

        print("VERTEX ERROR:", str(e))

        return (
            "Medical assistant temporarily unavailable."
        )
