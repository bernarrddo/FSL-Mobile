from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from model import ModifiedLSTM
import os
import base64
from io import BytesIO
from PIL import Image
import mediapipe as mp
import secrets
import time

app = Flask(__name__)
CORS(app)


# ---------------------------
# LOAD MODEL
# ---------------------------
MODEL_PATH = "run24.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    'Color_Black', 'Color_Blue', 'Color_Brown', 'Color_Dark', 'Color_Gray',
    'Color_Green', 'Color_Light', 'Color_Orange', 'Color_Pink', 'Color_Red',
    'Color_Violet', 'Color_White', 'Color_Yellow',
    'Family_Auntie', 'Family_Cousin', 'Family_Daughter', 'Family_Father',
    'Family_Grandfather', 'Family_Grandmother', 'Family_Mother', 'Family_Parents',
    'Family_Son', 'Family_Uncle',
    'Numbers_Eight', 'Numbers_Five', 'Numbers_Four', 'Numbers_Nine',
    'Numbers_One', 'Numbers_Seven', 'Numbers_Six', 'Numbers_Ten',
    'Numbers_Three', 'Numbers_Two'
]

# FIX: model input/output dimensions must match training
model = ModifiedLSTM(
    input_size=188,
    hidden_size=256,
    num_layers=2,
    num_classes=len(CLASSES)
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()


# ---------------------------
# MEDIAPIPE HAND EXTRACTION
# ---------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

def extract_landmarks(pil_image):
    image_rgb = np.array(pil_image.convert("RGB"))
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    # 21 * (x,y,z) = 63 values → pad to 188
    arr = []
    for lm in hand.landmark:
        arr.extend([lm.x, lm.y, lm.z])

    arr = np.array(arr, dtype=np.float32)

    # FIX: pad consistently
    if arr.shape[0] < 188:
        arr = np.pad(arr, (0, 188 - arr.shape[0]))
    else:
        arr = arr[:188]

    return arr

# ---------------------------
# PREDICTION ROUTE
# ---------------------------

@app.post("/predict")
def predict():
    body = request.json or {}

    frame_str = body.get("frame")
    if not frame_str:
        return jsonify({"error": "Missing frame"}), 400

    # FIX: handle format "data:image/jpeg;base64,XXXX"
    if "," in frame_str:
        frame_str = frame_str.split(",")[1]

    try:
        img = Image.open(BytesIO(base64.b64decode(frame_str)))
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    features = extract_landmarks(img)
    if features is None:
        return jsonify({"error": "No hand detected"}), 400

    x = torch.tensor(features, dtype=torch.float32).reshape(1, 1, 188).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = np.argmax(probs)

    return jsonify({
        "prediction": CLASSES[idx],
        "confidence": float(probs[idx]),
    })


@app.get("/ping")
def ping():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)