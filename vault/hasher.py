import hashlib
import os
import base64
import random

PEPPER_PATH = "data/pepper.secret"

# Load or create pepper
if not os.path.exists(PEPPER_PATH):
    with open(PEPPER_PATH, "wb") as f:
        f.write(os.urandom(16))

with open(PEPPER_PATH, "rb") as f:
    PEPPER = f.read()

def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    pwd_peppered = password.encode() + PEPPER
    hashed = hashlib.pbkdf2_hmac("sha256", pwd_peppered, salt, 100_000)
    return {
        "salt": base64.b64encode(salt).decode(),
        "hash": base64.b64encode(hashed).decode()
    }
