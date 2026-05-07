import os
import json
from dotenv import load_dotenv

from google.cloud import aiplatform

# ======================================================
# LOAD ENV VARIABLES
# ======================================================

load_dotenv()

# ======================================================
# CREATE GCP KEY FILE
# ======================================================

gcp_creds = os.getenv("GCP_CREDENTIALS")

if not gcp_creds:
    raise Exception("GCP_CREDENTIALS missing")

# Validate JSON
creds_dict = json.loads(gcp_creds)

# Save credentials file
with open("gcp_key.json", "w") as f:
    json.dump(creds_dict, f)

# Set environment variable
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
You are a helpful medical AI assistant.

Provide concise, accurate,
patient-friendly educational explanations.

Only provide educational information.

Do not provide diagnosis,
prescriptions,
or treatment plans.

Do not repeat the question.

Do not use markdown symbols
such as *, **, #.

Return clean plain text only.
"""

# ======================================================
# ASK MODEL
# ======================================================

def ask_biomistral(question: str):

    try:

        # ==================================================
        # PROMPT
        # ==================================================

        prompt = f"""
        {SYSTEM_PROMPT}

        Question:
        {question}

        Answer:
        """

        # ==================================================
        # REQUEST PAYLOAD
        # ==================================================

        instances = [
            {
                "prompt": prompt,

                "max_tokens": 700,

                "temperature": 0.6,

                "top_p": 0.9
            }
        ]

        # ==================================================
        # CALL VERTEX ENDPOINT
        # ==================================================

        response = endpoint.predict(
            instances=instances
        )

        print("VERTEX RESPONSE:", response)

        predictions = response.predictions

        # ==================================================
        # EMPTY RESPONSE
        # ==================================================

        if len(predictions) == 0:
            return "No response generated."

        pred = predictions[0]

        print("PRED:", pred)

        # ==================================================
        # STRING RESPONSE
        # ==================================================

        if isinstance(pred, str):

            text = pred

        # ==================================================
        # DICTIONARY RESPONSE
        # ==================================================

        elif isinstance(pred, dict):

            if "generated_text" in pred:
                text = pred["generated_text"]

            elif "content" in pred:
                text = pred["content"]

            elif "output" in pred:
                text = pred["output"]

            elif "response" in pred:
                text = pred["response"]

            else:
                text = str(pred)

        else:

            text = str(pred)

        # ==================================================
        # CLEAN RESPONSE
        # ==================================================

        unwanted_phrases = [

            "Prompt:",
            "Output:",
            "Assistant:",
            "Answer:",
            "Question:",
            SYSTEM_PROMPT,
            question
        ]

        for phrase in unwanted_phrases:
            text = text.replace(phrase, "")

        # --------------------------------------------------
        # REMOVE MARKDOWN SYMBOLS
        # --------------------------------------------------

        text = text.replace("**", "")
        text = text.replace("*", "")
        text = text.replace("###", "")
        text = text.replace("##", "")
        text = text.replace("#", "")

        # --------------------------------------------------
        # FIX ESCAPED NEWLINES
        # --------------------------------------------------

        text = text.replace("\\n", "\n")

        # --------------------------------------------------
        # REMOVE EXTRA EMPTY LINES
        # --------------------------------------------------

        lines = text.splitlines()

        cleaned_lines = []

        for line in lines:

            line = line.strip()

            if line != "":
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # --------------------------------------------------
        # FINAL CLEANUP
        # --------------------------------------------------

        text = text.strip()

        # ==================================================
        # RETURN CLEAN TEXT
        # ==================================================

        return text

    # ======================================================
    # ERROR HANDLING
    # ======================================================

    except Exception as e:

        print("VERTEX ERROR:", str(e))

        return (
            "Medical assistant temporarily unavailable."
        )