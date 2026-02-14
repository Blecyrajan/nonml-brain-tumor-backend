from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import shutil
import os
from datetime import datetime, timezone
from fastapi.staticfiles import StaticFiles
from llm_client import ask_biomistral



from database import users_collection, predictions_collection
from utils import hash_password, verify_password
from predict import predict_image

app = FastAPI()

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


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

# ---------------- PREDICT ----------------
from fastapi import Request
from datetime import datetime

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    user: str = ""
):
    print("PREDICT API HIT BY:", user)

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("IMAGE SAVED AT:", file_path)

    # 🔹 ML prediction
    result = predict_image(file_path)

    # 🔹 IMPORTANT: build image URL EXPLICITLY
    image_url = f"http://{request.headers['host']}/uploads/{file.filename}"

    print("IMAGE URL:", image_url)
    print("PREDICTION RESULT:", result)

    # 🔹 SAVE TO MONGODB (THIS WAS MISSING / WRONG)
    predictions_collection.insert_one({
        "user": user,
        "image": file.filename,
        "image_url": image_url,
        "prediction": result["class"],
        "confidence": result["confidence"],
        "timestamp": datetime.now(timezone.utc)
    })

    print("PREDICTION SAVED TO DB")

    # 🔹 RETURN EVERYTHING FLUTTER NEEDS
    return {
        "class": result["class"],
        "confidence": result["confidence"],
        "image_url": image_url
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
