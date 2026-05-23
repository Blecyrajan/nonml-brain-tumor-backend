import os
from dotenv import load_dotenv
import google.generativeai as genai

# ==========================================
# LOAD ENV VARIABLES
# ==========================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise Exception("GOOGLE_API_KEY missing")

# ==========================================
# CONFIGURE GEMINI
# ==========================================

genai.configure(api_key=API_KEY)

# ==========================================
# LOAD MODEL
# ==========================================

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are a helpful medical AI assistant
for a brain tumor detection application.

Provide:
- educational explanations
- patient-friendly answers
- concise responses

Rules:
- do NOT diagnose
- do NOT prescribe medicines
- do NOT provide treatment plans
- do NOT create fear
- do NOT repeat the question
- return clean plain text only
- no markdown symbols
"""

# ==========================================
# ASK GEMINI
# ==========================================

def ask_medical_llm(question, prediction=None):

    try:

        prediction_context = ""

        if prediction:

            prediction_context = f"""
            Predicted condition:
            {prediction}
            """

        prompt = f"""
        {SYSTEM_PROMPT}

        {prediction_context}

        User Question:
        {question}

        Assistant Response:
        """

        response = model.generate_content(
            prompt
        )

        text = response.text

        # CLEAN RESPONSE
        unwanted = [
            "Assistant Response:",
            "User Question:",
            question
        ]

        for item in unwanted:
            text = text.replace(item, "")

        text = text.replace("*", "")
        text = text.replace("#", "")

        return text.strip()

    except Exception as e:

        print("GEMINI ERROR:", str(e))

        return (
            "Medical assistant temporarily unavailable."
        )