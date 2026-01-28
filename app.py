from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pandas as pd
import cv2
import os
from datetime import datetime

# =========================
# APP CONFIG
# =========================
app = Flask(__name__)
CORS(app)  # allow cross-origin requests

IMG_SIZE = (224, 224)
MODEL_PATH = "model.h5"
CSV_PATH = "fertilizer.csv"

UPLOAD_FOLDER = "uploads"
CAMERA_FOLDER = "camera"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CAMERA_FOLDER, exist_ok=True)

# =========================
# LOAD MODEL & CSV
# =========================
model = load_model(MODEL_PATH, compile=False)
df = pd.read_csv(CSV_PATH)

CLASSES = [
    'Tomato___Target_Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Late_blight',
    'Tomato___Early_blight',
    'Tomato___healthy'
]

# =========================
# THRESHOLDS
# =========================
CONFIDENCE_THRESHOLD = 80
TOP_GAP_THRESHOLD = 12
GREEN_RATIO_THRESHOLD = 0.18

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return jsonify({"status": "Plant Disease Prediction API Running"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img_path = os.path.join(
        UPLOAD_FOLDER,
        datetime.now().strftime('%Y%m%d%H%M%S') + ".jpg"
    )
    file.save(img_path)
    return analyze_image(img_path)

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img_path = os.path.join(
        CAMERA_FOLDER,
        datetime.now().strftime('%Y%m%d%H%M%S') + ".jpg"
    )
    file.save(img_path)
    return analyze_image(img_path)

# =========================
# CORE LOGIC
# =========================
def analyze_image(img_path):
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        return invalid_response("Invalid image file", img_path)

    # Green leaf detection
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    if green_ratio < GREEN_RATIO_THRESHOLD:
        return invalid_response("Not a tomato leaf", img_path)

    # Model prediction
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)[0] * 100
    sorted_idx = np.argsort(preds)[::-1]

    best_idx = sorted_idx[0]
    second_idx = sorted_idx[1]

    best_conf = preds[best_idx]
    gap = best_conf - preds[second_idx]

    disease = CLASSES[best_idx]

    if best_conf < CONFIDENCE_THRESHOLD or gap < TOP_GAP_THRESHOLD:
        return invalid_response(
            "Unclear disease. Upload clear tomato leaf.", img_path
        )

    # CSV lookup
    row = df[df['disease'] == disease]
    if row.empty:
        return invalid_response("Disease information not found", img_path)

    response = {
        "disease": disease.replace("Tomato___", "").replace("_", " "),
        "confidence": round(float(best_conf), 2),
        "fertilizer": row['recommended_fertilizer'].values[0],
        "solution": row['solution'].values[0],
        "description": row['description'].values[0],
        "crop": row['crop'].values[0]
    }

    if os.path.exists(img_path):
        os.remove(img_path)

    return jsonify(response)

# =========================
# INVALID RESPONSE
# =========================
def invalid_response(message, img_path):
    if os.path.exists(img_path):
        os.remove(img_path)
    return jsonify({
        "disease": "Not a Tomato Leaf",
        "confidence": 0,
        "fertilizer": "N/A",
        "solution": message,
        "description": "Prediction requires a clear tomato leaf image.",
        "crop": "Tomato"
    }), 400

# =========================
# RUN APP (Render requirement)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
