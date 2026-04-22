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

#-----------FEATURE SCORES----------------
def compute_feature_scores(image_path, heatmap_path):
    # Load original image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))

    # Load heatmap
    heatmap = cv2.imread(heatmap_path)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    # Normalize heatmap
    heatmap_norm = heatmap_gray / 255.0

    # -----------------------------
    # 1. TUMOR REGION MASK
    # -----------------------------
    threshold = 0.6   # focus on important regions
    mask = heatmap_norm > threshold

    # Avoid empty mask
    if np.sum(mask) == 0:
        mask = heatmap_norm > 0.3

    # -----------------------------
    # 2. ASYMMETRY (ECL BASED)
    # -----------------------------
    left = img[:, :112]
    right = img[:, 112:]
    right_flipped = cv2.flip(right, 1)

    diff = cv2.absdiff(left, right_flipped)

    asymmetry_score = np.mean(diff * mask[:, :112])
    asymmetry_score = min(100, (asymmetry_score / 50) * 100)

    # -----------------------------
    # 3. TEXTURE (ONLY TUMOR REGION)
    # -----------------------------
    texture_score = np.std(img[mask])
    texture_score = min(100, (texture_score / 80) * 100)

    # -----------------------------
    # 4. BOUNDARY (EDGES IN REGION)
    # -----------------------------
    edges = cv2.Canny(img, 100, 200)
    boundary_score = np.mean(edges[mask])
    boundary_score = min(100, (boundary_score / 50) * 100)

    # -----------------------------
    # 5. TUMOR AREA %
    # -----------------------------
    tumor_area = (np.sum(mask) / (224 * 224)) * 100

    return {
        "asymmetry": round(asymmetry_score, 2),
        "texture": round(texture_score, 2),
        "boundary": round(boundary_score, 2),
        "tumor_area": round(tumor_area, 2)
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

    # Convert URL → local path if needed
    heatmap_path = file_path.replace(".jpg", "_heatmap.jpg")

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

