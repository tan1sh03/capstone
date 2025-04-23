import os
import json
import base64
from hasher import hash_password
from mutator import realistic_mutations
import random

REAL_PASSWORD_PATH = "data/real_password.txt"
OUTPUT_PATH = "data/honeytokens.json"

# Load the real password
with open(REAL_PASSWORD_PATH, "r", encoding="utf-8") as f:
    real_password = f.read().strip()

# Mutate variants
decoys = realistic_mutations(real_password, count=20)

# Include the real password at a random position
all_passwords = decoys + [real_password, real_password]  # Adding twice to increase difficulty
random.shuffle(all_passwords)

# Create honeytoken vault
vault = []
for pwd in all_passwords:
    hash_val, salt = hash_password(pwd)
    vault.append({
        "plain": pwd,
        "hash": hash_val,
        "salt": salt,
        "is_real": pwd == real_password
    })

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(vault, f, indent=2)

print(f"\n✅ Generated Semantic Honeytokens saved at: {OUTPUT_PATH}")
