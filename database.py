import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)

# Create database
db = client["brain_tumor_app"]

# Collections
users_collection = db["users"]
predictions_collection = db["predictions"]
