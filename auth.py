import bcrypt
from database import users

def register(email, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({"email": email, "password": hashed})

def login(email, password):
    user = users.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        return True
    return False
