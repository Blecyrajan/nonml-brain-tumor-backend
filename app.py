#backend/app.py
from fileinput import filename
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import shutil
import os
import time
from datetime import datetime, timezone
from fastapi.staticfiles import StaticFiles
from llm_client import ask_biomistral
import requests
from database import users_collection, predictions_collection
from utils import hash_password, verify_password
import random
import cv2
import numpy as np

app = FastAPI()

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

HF_PREDICT_URL = "https://blecy2002-brain-tumor-predictor.hf.space/predict"


# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ChatRequest(BaseModel):
    user: EmailStr
    question: str

# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"status": "Backend running"}

# ---------------- REGISTER ----------------
@app.post("/register")
def register_user(data: RegisterRequest):

    print("REGISTER API HIT:", data.email)

    if users_collection.find_one({"email": data.email}):
        raise HTTPException(status_code=400, detail="User already exists")

    users_collection.insert_one({
        "email": data.email,
        "password": hash_password(data.password)
    })

    print("USER INSERTED INTO DB")

    return {"message": "User registered successfully"}

# ---------------- LOGIN ----------------
@app.post("/login")
def login_user(data: LoginRequest):

    user = users_collection.find_one({"email": data.email})

    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "message": "Login successful",
        "email": user["email"]
    }

def compute_feature_scores(image_path, heatmap_path):
    if not os.path.exists(heatmap_path):
        heatmap_path = image_path

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))

    # Load heatmap
    heatmap = cv2.imread(heatmap_path)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    heatmap_norm = heatmap_gray / 255.0

    # -----------------------------
    # Tumor Mask
    # -----------------------------
    mask = heatmap_norm > 0.5
    if np.sum(mask) == 0:
        mask = heatmap_norm > 0.3

    # -----------------------------
    # 1. Tumor Size (%)
    # -----------------------------
    tumor_area = (np.sum(mask) / (224 * 224)) * 100

    # -----------------------------
    # 2. Tumor Location
    # -----------------------------
    left_area = np.sum(mask[:, :112])
    right_area = np.sum(mask[:, 112:])

    if left_area > right_area * 1.2:
        location = "Left Hemisphere"
    elif right_area > left_area * 1.2:
        location = "Right Hemisphere"
    else:
        location = "Central"

    # -----------------------------
    # 3. Shape Irregularity
    # -----------------------------
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)

        if area > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
            irregularity = (1 - circularity) * 100
        else:
            irregularity = 0
    else:
        irregularity = 0

    irregularity = min(100, irregularity)

    # -----------------------------
    # 4. Intensity Heterogeneity
    # -----------------------------
    tumor_pixels = img[mask]

    if len(tumor_pixels) > 0:
        heterogeneity = np.std(tumor_pixels)
        heterogeneity = min(100, (heterogeneity / 80) * 100)
    else:
        heterogeneity = 0

    return {
        "tumor_size_%": round(tumor_area, 2),
        "tumor_location": location,
        "shape_irregularity_%": round(irregularity, 2),
        "heterogeneity_%": round(heterogeneity, 2)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), user: str = Form(...)):
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)

    # Create unique filename
    filename = f"{user}_{int(time.time())}_{file.filename}"
    file_path = f"uploads/{filename}"

    # Save uploaded image locally
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Call Hugging Face Space
    with open(file_path, "rb") as f:
        response = requests.post(
            HF_PREDICT_URL,
            files={"file": f}
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Prediction service failed")

    result = response.json()


    # Build full image URL
    BASE_URL = "https://nonml-brain-tumor-backend.onrender.com"
    image_url = f"{BASE_URL}/uploads/{filename}"

    heatmap_url = result.get("heatmap_url", "")

    # default fallback
    heatmap_path = file_path

    # If HF returned heatmap URL, download locally
    if heatmap_url:
        local_heatmap_path = file_path.replace(".jpg", "_heatmap.jpg").replace(".png", "_heatmap.jpg").replace(".jpeg", "_heatmap.jpg")

        try:
            r = requests.get(heatmap_url, timeout=20)

            if r.status_code == 200:
                with open(local_heatmap_path, "wb") as f:
                    f.write(r.content)

                heatmap_path = local_heatmap_path

        except Exception as e:
            print("Heatmap download failed:", e)

    features = compute_feature_scores(file_path, heatmap_path)

    # Save prediction to MongoDB
    predictions_collection.insert_one({
        "user": user,
        "prediction": result["class"],
        "confidence": result["confidence"],
        "image_url": image_url,
        "timestamp": datetime.now(timezone.utc)
    })

    # return image_url to frontend
    return {
        "class": result["class"],
        "confidence": result["confidence"],
        "image_url": image_url,
        "heatmap_url": heatmap_url,
        "features": features
    }


# ---------------- HISTORY ----------------

@app.get("/history")
def get_history(user: str):
    records = predictions_collection.find(
        {"user": user},
        {"_id": 0}
    ).sort("timestamp", -1)

    return list(records)

# ---------------- CHAT WITH BIOMISTRAL ----------------
@app.post("/chat")
def chat_with_ai(data: ChatRequest):
    print("CHAT REQUEST FROM:", data.user)
    answer = ask_biomistral(data.question)
    return {"answer": answer}

