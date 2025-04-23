import time
import pandas as pd
from pynput import keyboard
import os

# Define output file path
OUTPUT_DIR = "datasets/keystroke/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "user_training_keystrokes.csv")

# Ensure directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample text for training
SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence contains almost every letter. "
    "Keystroke dynamics help detect unauthorized access."
)

# Initialize keystroke data storage
keystroke_data = []
pressed_keys = {}

print("\n🎯 Type the following text exactly as shown below:")
print("------------------------------------------------------")
print(SAMPLE_TEXT)
print("------------------------------------------------------")
print("💡 Press Enter once you've finished typing the full text.\n")

# Callback for key press
def on_press(key):
    try:
        key_name = key.char if hasattr(key, 'char') else str(key)
        pressed_keys[key_name] = time.time()
    except Exception as e:
        print(f"❌ Error processing key press: {e}")

# Callback for key release
def on_release(key):
    try:
        key_name = key.char if hasattr(key, 'char') else str(key)
        if key_name in pressed_keys:
            press_time = pressed_keys[key_name]
            release_time = time.time()
            hold_time = release_time - press_time

            # Record flight time if there's a previous key
            if keystroke_data:
                last_key_time = keystroke_data[-1][1]
                flight_time = press_time - last_key_time
            else:
                flight_time = 0  # First key has no flight time

            keystroke_data.append((key_name, press_time, release_time, hold_time, flight_time))

        if key == keyboard.Key.enter:
            return False
    except Exception as e:
        print(f"❌ Error processing key release: {e}")

# Start recording
print("📝 Start typing now...\n")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# Convert collected data to DataFrame
df_new = pd.DataFrame(keystroke_data, columns=["key", "press_time", "release_time", "hold_time", "flight_time"])

# Append to existing dataset (if it exists)
if os.path.exists(OUTPUT_FILE):
    df_existing = pd.read_csv(OUTPUT_FILE)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_final = df_new

# Save the updated dataset
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Keystroke data appended to: {OUTPUT_FILE}")
