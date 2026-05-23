from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

import os
import time
import requests
import cv2
import numpy as np

from datetime import datetime, timezone

from llm_client import ask_medical_llm
from database import users_collection, predictions_collection
from utils import hash_password, verify_password

import cloudinary
from cloudinary import uploader


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()


# =========================================================
# CLOUDINARY CONFIG
# =========================================================

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


# =========================================================
# HUGGING FACE MODEL API
# =========================================================

HF_PREDICT_URL = "https://blecy2002-brain-tumor-predictor.hf.space/predict"


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODELS
# =========================================================

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChatRequest(BaseModel):
    user: EmailStr
    question: str
    prediction: str | None = None


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {"status": "Backend running"}


# =========================================================
# REGISTER
# =========================================================

@app.post("/register")
def register_user(data: RegisterRequest):

    print("REGISTER API HIT:", data.email)

    if users_collection.find_one({"email": data.email}):
        raise HTTPException(
            status_code=400,
            detail="User already exists"
        )

    users_collection.insert_one({
        "email": data.email,
        "password": hash_password(data.password)
    })

    return {
        "message": "User registered successfully"
    }


# =========================================================
# LOGIN
# =========================================================

@app.post("/login")
def login_user(data: LoginRequest):

    user = users_collection.find_one({
        "email": data.email
    })

    if not user or not verify_password(
        data.password,
        user["password"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    return {
        "message": "Login successful",
        "email": user["email"]
    }


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def compute_feature_scores(image_path, heatmap_path):

    if not os.path.exists(heatmap_path):
        heatmap_path = image_path

    # =============================
    # LOAD MRI
    # =============================

    img = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    img = cv2.resize(img, (224, 224))

    # =============================
    # LOAD HEATMAP
    # =============================

    heatmap = cv2.imread(heatmap_path)

    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap_gray = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2GRAY
    )

    heatmap_norm = heatmap_gray / 255.0

    # =============================
    # TUMOR MASK
    # =============================

    mask = heatmap_norm > 0.5

    if np.sum(mask) == 0:
        mask = heatmap_norm > 0.3

    # =============================
    # 1. TUMOR SIZE
    # =============================

    tumor_area = (
        np.sum(mask) / (224 * 224)
    ) * 100

    # =============================
    # 2. LOCATION
    # =============================

    left_area = np.sum(mask[:, :112])
    right_area = np.sum(mask[:, 112:])

    if left_area > right_area * 1.2:
        location = "Left Hemisphere"

    elif right_area > left_area * 1.2:
        location = "Right Hemisphere"

    else:
        location = "Central"

    # =============================
    # 3. SHAPE IRREGULARITY
    # =============================

    mask_uint8 = (
        mask * 255
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    irregularity = 0

    if len(contours) > 0:

        cnt = max(
            contours,
            key=cv2.contourArea
        )

        perimeter = cv2.arcLength(
            cnt,
            True
        )

        area = cv2.contourArea(cnt)

        if area > 0:

            circularity = (
                4 * np.pi * area
            ) / (perimeter ** 2 + 1e-8)

            irregularity = (
                1 - circularity
            ) * 100

    irregularity = min(100, irregularity)

    # =============================
    # 4. HETEROGENEITY
    # =============================

    tumor_pixels = img[mask]

    if len(tumor_pixels) > 0:

        heterogeneity = np.std(tumor_pixels)

        heterogeneity = min(
            100,
            (heterogeneity / 80) * 100
        )

    else:
        heterogeneity = 0

    # =============================
    # RETURN FEATURES
    # =============================

    return {
        "tumor_size_%": round(tumor_area, 2),
        "tumor_location": location,
        "shape_irregularity_%": round(irregularity, 2),
        "heterogeneity_%": round(heterogeneity, 2)
    }


# =========================================================
# PREDICTION API
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user: str = Form(...)
):

    os.makedirs("uploads", exist_ok=True)

    # =============================
    # SAVE MRI TEMPORARILY
    # =============================

    filename = (
        f"{user}_{int(time.time())}_{file.filename}"
    )

    file_path = f"uploads/{filename}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # =============================
    # CALL HUGGING FACE MODEL
    # =============================

    with open(file_path, "rb") as f:

        response = requests.post(
            HF_PREDICT_URL,
            files={"file": f}
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=500,
            detail="Prediction service failed"
        )

    result = response.json()

    # =============================
    # UPLOAD MRI TO CLOUDINARY
    # =============================

    upload_result = cloudinary.uploader.upload(
        file_path
    )

    image_url = upload_result["secure_url"]

    # =============================
    # DOWNLOAD HEATMAP
    # =============================

    hf_heatmap_url = result.get(
        "heatmap_url",
        ""
    )

    local_heatmap_path = ""
    heatmap_url = ""

    if hf_heatmap_url:

        local_heatmap_path = (
            file_path
            .replace(".jpg", "_heatmap.jpg")
            .replace(".png", "_heatmap.jpg")
            .replace(".jpeg", "_heatmap.jpg")
        )

        try:

            r = requests.get(
                hf_heatmap_url,
                timeout=20
            )

            if r.status_code == 200:

                with open(
                    local_heatmap_path,
                    "wb"
                ) as f:

                    f.write(r.content)

                # =============================
                # UPLOAD HEATMAP
                # =============================

                heatmap_upload = (
                    cloudinary.uploader.upload(
                        local_heatmap_path
                    )
                )

                heatmap_url = (
                    heatmap_upload["secure_url"]
                )

        except Exception as e:
            print(
                "Heatmap download failed:",
                e
            )

    # =============================
    # COMPUTE FEATURES
    # =============================

    features = compute_feature_scores(
        file_path,
        local_heatmap_path
        if local_heatmap_path
        else file_path
    )

    # =============================
    # SAVE HISTORY TO MONGODB
    # =============================

    predictions_collection.insert_one({

        "user": user,

        "prediction": result["class"],

        "confidence": result["confidence"],

        "image_url": image_url,

        "heatmap_url": heatmap_url,

        "features": features,

        "timestamp": datetime.now(
            timezone.utc
        ).strftime("%d %b %Y, %I:%M %p")
    })

    # =============================
    # CLEAN TEMP FILES
    # =============================

    try:

        if os.path.exists(file_path):
            os.remove(file_path)

        if (
            local_heatmap_path
            and os.path.exists(local_heatmap_path)
        ):
            os.remove(local_heatmap_path)

    except Exception as e:
        print("Cleanup error:", e)

    # =============================
    # RETURN RESPONSE
    # =============================

    return {
        "class": result["class"],
        "confidence": result["confidence"],
        "image_url": image_url,
        "heatmap_url": heatmap_url,
        "features": features
    }


# =========================================================
# HISTORY
# =========================================================

@app.get("/history")
def get_history(user: str):

    records = predictions_collection.find(
        {"user": user},
        {"_id": 0}
    ).sort("timestamp", -1)

    return list(records)


# =========================================================
# CHATBOT
# =========================================================

@app.post("/chat")
def chat_with_ai(data: ChatRequest):

    print("CHAT REQUEST FROM:", data.user)

    answer = ask_medical_llm(
        data.question,
        data.prediction
    )

    return {
        "answer": answer
    }