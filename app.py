"""
app.py
------
AI-Based Crop Recommendation System — Flask entry point.

This is the production entry point used both locally (`python app.py`)
and on the cloud (via `gunicorn app:app`, see Procfile).

Design notes for cloud-readiness:
- All file paths are built from BASE_DIR (this file's own folder) using
  os.path.join, so there is NO dependency on local Windows paths like
  C:\\Users\\... The app runs identically on Windows, Linux, macOS, or
  any cloud host's container filesystem.
- The port is read from the PORT environment variable (falls back to
  5000 locally). Cloud platforms like Render inject PORT automatically.
- Model, scaler, and label encoder are loaded once at startup and kept
  in memory — no per-request disk reads.
- CORS is enabled so the frontend (even if hosted separately in the
  future) can call the /predict API without cross-origin errors.
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths — all relative to this file's own directory. No hardcoded local
# paths anywhere, so this works the same on any machine or cloud host.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

app = Flask(__name__)
CORS(app)  # allow cross-origin requests to the API endpoints

# ---------------------------------------------------------------------------
# Load model artifacts ONCE at startup.
# If any file is missing, fail loudly and early with a clear message
# instead of crashing confusingly on the first prediction request.
# ---------------------------------------------------------------------------
model = None
scaler = None
label_encoder = None
_load_error = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Run 'python model/train_model.py' first to generate it."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("[startup] Model, scaler, and label encoder loaded successfully.")
except Exception as e:
    _load_error = str(e)
    print(f"[startup] ERROR loading model artifacts: {_load_error}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple health-check endpoint — useful for verifying cloud deployment."""
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({"status": status, "error": _load_error}), (200 if model else 500)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "success": False,
            "error": f"Model not loaded on server: {_load_error}"
        }), 500

    # Accept both JSON body (API/JS fetch) and form submissions
    data = request.get_json(silent=True) or request.form

    try:
        values = []
        for feat in FEATURE_ORDER:
            if feat not in data or data[feat] in (None, ""):
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {feat}"
                }), 400
            values.append(float(data[feat]))
    except (ValueError, TypeError) as e:
        return jsonify({
            "success": False,
            "error": f"Invalid numeric value: {e}"
        }), 400

    try:
        import pandas as pd
        X = pd.DataFrame([values], columns=FEATURE_ORDER)
        X_scaled = scaler.transform(X)
        pred_idx = model.predict(X_scaled)[0]
        crop_name = label_encoder.inverse_transform([pred_idx])[0]

        # Top-3 probabilities for a nicer, more informative UI
        probs = model.predict_proba(X_scaled)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {"crop": label_encoder.inverse_transform([i])[0], "confidence": round(float(probs[i]) * 100, 2)}
            for i in top3_idx
        ]

        return jsonify({
            "success": True,
            "recommended_crop": crop_name,
            "top_3": top3
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {e}"}), 500


# ---------------------------------------------------------------------------
# Local dev entry point.
# In production, gunicorn imports the `app` object directly (see Procfile)
# and this block is never executed — but it's kept so `python app.py`
# still works fine for local testing.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
