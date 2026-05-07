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

with open("gcp_key.json", "w") as f:
    f.write(gcp_creds)

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
# ASK MODEL
# ======================================================

def ask_biomistral(question: str):

    try:

        instances = [
            {
                "inputs": question
            }
        ]

        response = endpoint.predict(
            instances=instances
        )

        print("VERTEX RESPONSE:", response)

        # ==========================================
        # HANDLE RESPONSE
        # ==========================================

        predictions = response.predictions

        if len(predictions) == 0:
            return "No response generated."

        pred = predictions[0]

        # string response
        if isinstance(pred, str):
            return pred

        # dict response
        if isinstance(pred, dict):

            if "generated_text" in pred:
                return pred["generated_text"]

            if "content" in pred:
                return pred["content"]

            if "outputs" in pred:
                return pred["outputs"]

            return str(pred)

        return str(pred)

    except Exception as e:

        print("VERTEX ERROR:", str(e))

        return (
            "Medical assistant temporarily unavailable."
        )