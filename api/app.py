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
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pearls AQI Predictor REST API</title>
            <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
            <style>
                :root {
                    --bg: #0B0F19;
                    --panel: rgba(22, 30, 49, 0.7);
                    --border: rgba(255, 255, 255, 0.08);
                    --text: #F8FAFC;
                    --text-muted: #94A3B8;
                    --accent-blue: #38BDF8;
                    --accent-purple: #818CF8;
                    --accent-orange: #F59E0B;
                    --green: #10B981;
                }
                body {
                    font-family: 'Plus Jakarta Sans', sans-serif;
                    background-color: var(--bg);
                    color: var(--text);
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    box-sizing: border-box;
                }
                .container {
                    width: 100%;
                    max-width: 800px;
                    padding: 40px 20px;
                }
                .header {
                    text-align: center;
                    margin-bottom: 40px;
                }
                .badge {
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    background: rgba(16, 185, 129, 0.1);
                    color: var(--green);
                    border: 1px solid rgba(16, 185, 129, 0.2);
                    padding: 6px 16px;
                    border-radius: 99px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    margin-bottom: 16px;
                }
                .badge-dot {
                    width: 8px;
                    height: 8px;
                    background: var(--green);
                    border-radius: 50%;
                    box-shadow: 0 0 10px var(--green);
                }
                h1 {
                    font-size: 2.5rem;
                    font-weight: 800;
                    margin: 0 0 10px 0;
                    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                .subtitle {
                    color: var(--text-muted);
                    font-size: 1.1rem;
                    margin: 0;
                }
                .card {
                    background: var(--panel);
                    border: 1px solid var(--border);
                    border-radius: 20px;
                    padding: 30px;
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
                    backdrop-filter: blur(16px);
                    margin-bottom: 24px;
                }
                h3 {
                    margin-top: 0;
                    font-size: 1.25rem;
                    font-weight: 600;
                    border-bottom: 1px solid var(--border);
                    padding-bottom: 12px;
                    margin-bottom: 20px;
                    color: var(--accent-blue);
                }
                ul {
                    list-style: none;
                    padding: 0;
                    margin: 0;
                }
                li {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 16px 0;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
                }
                li:last-child {
                    border-bottom: none;
                }
                .endpoint-info {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }
                .endpoint-title {
                    font-weight: 600;
                    font-size: 1rem;
                }
                .endpoint-desc {
                    font-size: 0.85rem;
                    color: var(--text-muted);
                }
                .btn {
                    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
                    color: var(--text);
                    text-decoration: none;
                    padding: 10px 20px;
                    border-radius: 12px;
                    font-size: 0.9rem;
                    font-weight: 600;
                    transition: transform 0.2s, box-shadow 0.2s;
                    box-shadow: 0 4px 15px rgba(129, 140, 248, 0.2);
                }
                .btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(129, 140, 248, 0.4);
                }
                .code-block {
                    background: #07090E;
                    border: 1px solid var(--border);
                    border-radius: 12px;
                    padding: 16px;
                    font-family: monospace;
                    font-size: 0.85rem;
                    color: var(--accent-orange);
                    overflow-x: auto;
                    margin-top: 10px;
                }
                .footer {
                    text-align: center;
                    font-size: 0.8rem;
                    color: var(--text-muted);
                    margin-top: 40px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="badge">
                        <span class="badge-dot"></span>
                        Service Online
                    </div>
                    <h1>Pearls AQI REST API Gateway</h1>
                    <p class="subtitle">Islamabad AQI 3-Day ML Forecasting Service</p>
                </div>
                
                <div class="card">
                    <h3>⚡ Available Endpoints</h3>
                    <ul>
                        <li>
                            <div class="endpoint-info">
                                <span class="endpoint-title">Health Check</span>
                                <span class="endpoint-desc">Verify server status and active loaded models.</span>
                            </div>
                            <a href="/health" class="btn">Query /health</a>
                        </li>
                        <li>
                            <div class="endpoint-info">
                                <span class="endpoint-title">Get 3-Day AQI Forecast</span>
                                <span class="endpoint-desc">Retrieves the latest prediction dataset.</span>
                            </div>
                            <a href="/predict" class="btn">Query /predict</a>
                        </li>
                    </ul>
                </div>

                <div class="card">
                    <h3>📖 API Usage Quickstart</h3>
                    <p class="endpoint-desc">Send a standard HTTP GET request to forecast the 3-day AQI. Defaults to Islamabad weather parameters.</p>
                    <div class="code-block">
                        curl -X GET http://localhost:5000/predict?city=Islamabad
                    </div>
                </div>

                <div class="footer">
                    Pearls AQI Predictor &copy; 2026 | Serverless MLOps Platform
                </div>
            </div>
        </body>
    </html>
    """, 200

def render_json_or_html(data, status_code=200):
    """
    Renders beautiful HTML JSON syntax viewer if requested from browser,
    otherwise returns raw JSON response.
    """
    accept_header = request.headers.get('Accept', '')
    if 'text/html' in accept_header:
        import json
        pretty_json = json.dumps(data, indent=4)
        return f"""
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>API JSON Response Viewer</title>
                <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
                <style>
                    :root {{
                        --bg: #0B0F19;
                        --panel: rgba(22, 30, 49, 0.7);
                        --border: rgba(255, 255, 255, 0.08);
                        --text: #F8FAFC;
                        --text-muted: #94A3B8;
                        --accent-blue: #38BDF8;
                        --accent-purple: #818CF8;
                        --green: #10B981;
                    }}
                    body {{
                        font-family: 'Plus Jakarta Sans', sans-serif;
                        background-color: var(--bg);
                        color: var(--text);
                        margin: 0;
                        padding: 40px 20px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        box-sizing: border-box;
                    }}
                    .card {{
                        background: var(--panel);
                        border: 1px solid var(--border);
                        border-radius: 20px;
                        padding: 30px;
                        width: 100%;
                        max-width: 750px;
                        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
                        backdrop-filter: blur(16px);
                    }}
                    h2 {{
                        margin-top: 0;
                        font-size: 1.4rem;
                        font-weight: 700;
                        color: var(--accent-blue);
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }}
                    .status {{
                        font-size: 0.8rem;
                        background: rgba(16, 185, 129, 0.1);
                        color: var(--green);
                        border: 1px solid rgba(16, 185, 129, 0.2);
                        padding: 4px 12px;
                        border-radius: 8px;
                    }}
                    pre {{
                        background: #07090E;
                        border: 1px solid var(--border);
                        border-radius: 12px;
                        padding: 20px;
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 0.9rem;
                        color: #F59E0B;
                        overflow-x: auto;
                        line-height: 1.5;
                        margin: 20px 0;
                    }}
                    .back-btn {{
                        display: inline-block;
                        color: var(--text-muted);
                        text-decoration: none;
                        font-size: 0.85rem;
                        font-weight: 600;
                        transition: color 0.2s;
                    }}
                    .back-btn:hover {{
                        color: var(--accent-blue);
                    }}
                </style>
            </head>
            <body>
                <div class="card">
                    <h2>
                        <span>⚡ JSON Response Viewer</span>
                        <span class="status">HTTP {status_code} OK</span>
                    </h2>
                    <pre>{pretty_json}</pre>
                    <a href="/" class="back-btn">&larr; Return to API Gateway</a>
                </div>
            </body>
        </html>
        """, status_code
    return jsonify(data), status_code

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return render_json_or_html({
        "status": "ok",
        "service": "Pearls AQI Predictor REST API",
        "city": os.getenv("CITY", "Islamabad"),
        "models_loaded": {
            "random_forest": rf_model is not None,
            "tensorflow": tf_model is not None,
            "scaler": scaler is not None
        }
    }, 200)

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
            return render_json_or_html({
                "status": "error",
                "message": "No functional prediction model available."
            }, 500)

        forecast_24h = max(0.0, float(predictions[0]))
        forecast_48h = max(0.0, float(predictions[1]))
        forecast_72h = max(0.0, float(predictions[2]))

        risk_level = get_risk_level(current_aqi)

        return render_json_or_html({
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
        }, 200)

    except Exception as e:
        return render_json_or_html({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }, 500)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
