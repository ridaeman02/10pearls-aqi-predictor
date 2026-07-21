import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# Ensure workspace root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.feature_store import get_feature_store

app = Flask(__name__)

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model objects
rf_model = None
tf_model = None
scaler = None
feature_names = None

def get_risk_level(aqi_val: float) -> str:
    """Returns official EPA AQI risk category."""
    val = round(aqi_val)
    if val <= 50:
        return "Good"
    elif val <= 100:
        return "Moderate"
    elif val <= 150:
        return "Unhealthy for Sensitive Groups"
    elif val <= 200:
        return "Unhealthy"
    elif val <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def load_artifacts():
    """Load model binaries, scaler, and feature metadata from models/ directory."""
    global rf_model, tf_model, scaler, feature_names
    
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    tf_path = os.path.join(MODELS_DIR, "tf_aqi_model.h5")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    # Load Scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # Load Feature Names
    if os.path.exists(features_path):
        with open(features_path, "rb") as f:
            feature_names = pickle.load(f)
    else:
        feature_names = [
            'pm25', 'pm10', 'no2', 'temp', 'humidity', 'wind_speed', 'pressure',
            'hour', 'day_of_week', 'month', 'aqi', 'aqi_change_rate'
        ]

    # Load Random Forest Model
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)

    # Load TensorFlow Model
    if os.path.exists(tf_path):
        try:
            import tensorflow as tf
            tf_model = tf.keras.models.load_model(tf_path)
        except (ImportError, Exception) as e:
            print(f"Warning: Could not load TensorFlow model: {e}")

# Initial load on startup
load_artifacts()

@app.route("/", methods=["GET"])
def home():
    """Default landing route helper."""
    return """
    <html>
        <head>
            <title>Pearls AQI Predictor API</title>
            <style>
                body { font-family: sans-serif; margin: 40px; background: #0F172A; color: #E2E8F0; }
                h1 { color: #38BDF8; }
                a { color: #F59E0B; text-decoration: none; font-weight: bold; }
                a:hover { text-decoration: underline; }
                .card { background: #1E293B; padding: 20px; border-radius: 8px; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>🌤️ Pearls AQI Predictor REST API</h1>
            <p>Welcome! The server is running successfully.</p>
            <div class="card">
                <h3>Available Endpoints:</h3>
                <ul>
                    <li><strong>Health Check:</strong> <a href="/health">/health</a></li>
                    <li><strong>Forecast Predictions:</strong> <a href="/predict">/predict</a></li>
                </ul>
            </div>
        </body>
    </html>
    """, 200

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "Pearls AQI Predictor REST API",
        "city": os.getenv("CITY", "Islamabad"),
        "models_loaded": {
            "random_forest": rf_model is not None,
            "tensorflow": tf_model is not None,
            "scaler": scaler is not None
        }
    }), 200

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """3-Day AQI Forecast Prediction Endpoint."""
    global rf_model, tf_model, scaler, feature_names
    
    # Reload artifacts if not loaded
    if scaler is None or (rf_model is None and tf_model is None):
        load_artifacts()

    if scaler is None:
        return jsonify({
            "status": "error",
            "message": "Model artifacts or scaler not found in models/ directory. Please run pipelines/training_pipeline.py first."
        }), 500

    try:
        # Determine city target
        city = request.args.get("city") if request.method == "GET" else None
        if not city and request.is_json:
            city = request.json.get("city")
        if not city:
            city = os.getenv("CITY", "Islamabad")

        # Pull newest feature vector from Feature Store
        fs = get_feature_store()
        aqi_fg = fs.get_feature_group(name="aqi_predictions_fg", version=1)
        df = aqi_fg.read()

        if df is None or df.empty:
            return jsonify({
                "status": "error",
                "message": "Feature Store is empty. Please run backfill or feature pipelines first."
            }), 404

        # Filter by city and sort to get newest vector
        city_df = df[df['city'].str.lower() == city.lower()] if 'city' in df.columns else df
        if city_df.empty:
            city_df = df # Fallback to available records

        latest_record = city_df.sort_values(by='timestamp', ascending=False).iloc[0]
        current_aqi = float(latest_record.get('aqi', 50))
        record_timestamp = str(latest_record.get('timestamp', ''))

        # Prepare feature vector
        vector = []
        for col in feature_names:
            val = float(latest_record.get(col, 0.0))
            vector.append(val)

        X_input = np.array([vector])
        X_scaled = scaler.transform(X_input)

        # Run Prediction (Primary: TensorFlow, Fallback: Random Forest)
        model_used = "TensorFlow Deep Learning Model"
        predictions = None

        if tf_model is not None:
            try:
                preds = tf_model.predict(X_scaled, verbose=0)
                predictions = preds[0]
            except Exception as e:
                print(f"TensorFlow prediction failed ({e}), falling back to Random Forest.")

        if predictions is None and rf_model is not None:
            model_used = "Random Forest Regressor"
            preds = rf_model.predict(X_scaled)
            predictions = preds[0]

        if predictions is None:
            return jsonify({
                "status": "error",
                "message": "No functional prediction model available."
            }), 500

        forecast_24h = max(0.0, float(predictions[0]))
        forecast_48h = max(0.0, float(predictions[1]))
        forecast_72h = max(0.0, float(predictions[2]))

        risk_level = get_risk_level(current_aqi)

        return jsonify({
            "status": "success",
            "city": city,
            "timestamp": record_timestamp,
            "current_aqi": round(current_aqi, 2),
            "risk_level": risk_level,
            "forecast": {
                "24h": round(forecast_24h, 2),
                "48h": round(forecast_48h, 2),
                "72h": round(forecast_72h, 2)
            },
            "model_used": model_used
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
