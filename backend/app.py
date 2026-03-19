"""
app.py - Flask backend for the Crop Recommendation System
==========================================================
Loads the trained model and exposes a POST /predict endpoint.

Usage:
    python backend/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding="utf-8")

# Add the backend directory to the path so we can import utils
sys.path.insert(0, os.path.dirname(__file__))
from utils import validate_input, REQUIRED_FIELDS

# -----------------------------------------------
# Initialise Flask app
# -----------------------------------------------
app = Flask(__name__)
CORS(app)  # allow cross-origin requests from the Streamlit frontend

# -----------------------------------------------
# Load the trained model at startup
# -----------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model.pkl")
MODEL_PATH = os.path.abspath(MODEL_PATH)

try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"[ERROR] Model file not found at {MODEL_PATH}")
    print("   Please run  python train_model.py  first.")
    sys.exit(1)


# -----------------------------------------------
# Routes
# -----------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Health-check endpoint."""
    return jsonify({"status": "ok", "message": "Crop Recommendation API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the best crop based on soil & weather features.

    Expects JSON body:
    {
        "N": <number>,
        "P": <number>,
        "K": <number>,
        "temperature": <number>,
        "humidity": <number>,
        "ph": <number>,
        "rainfall": <number>
    }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON in request body."}), 400

    # Validate input
    is_valid, error_msg = validate_input(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    # Build feature DataFrame in the order the model was trained on
    # Using a DataFrame preserves column names and avoids sklearn warnings
    features = pd.DataFrame([[
        data["N"],
        data["P"],
        data["K"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"],
    ]], columns=["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"])

    try:
        prediction = model.predict(features)
        return jsonify({"recommended_crop": prediction[0]})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# -----------------------------------------------
# Run the server
# -----------------------------------------------
if __name__ == "__main__":
    print("[INFO] Starting Crop Recommendation API on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
