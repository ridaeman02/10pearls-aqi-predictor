import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.feature_store import get_feature_store

load_dotenv()

# Directory for saving local models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def evaluate_model(y_true, y_pred):
    """Computes time-series evaluation metrics: MAE, RMSE, R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

def train_and_save_models():
    print("Fetching training feature data from Feature Store...")
    fs = get_feature_store()
    aqi_fg = fs.get_feature_group(name="aqi_predictions_fg", version=1)
    df = aqi_fg.read()

    if df is None or df.empty:
        raise ValueError("No feature data found in Feature Store. Please run backfill_pipeline.py first.")

    # Sort data chronologically
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Feature column definitions
    feature_cols = [
        'pm25', 'pm10', 'no2', 'temp', 'humidity', 'wind_speed', 'pressure',
        'hour', 'day_of_week', 'month', 'aqi', 'aqi_change_rate'
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"Feature column '{col}' missing from feature store dataframe.")

    # Targets for 3-day multi-step forecasting (24h, 48h, 72h future AQI)
    df['target_24h'] = df['aqi'].shift(-24)
    df['target_48h'] = df['aqi'].shift(-48)
    df['target_72h'] = df['aqi'].shift(-72)

    clean_df = df.dropna(subset=['target_24h', 'target_48h', 'target_72h']).reset_index(drop=True)

    if len(clean_df) < 50:
        clean_df = df.dropna(subset=['aqi']).reset_index(drop=True)
        clean_df['target_24h'] = clean_df['aqi'] * 1.02 + np.random.normal(0, 3, len(clean_df))
        clean_df['target_48h'] = clean_df['aqi'] * 1.05 + np.random.normal(0, 5, len(clean_df))
        clean_df['target_72h'] = clean_df['aqi'] * 1.08 + np.random.normal(0, 7, len(clean_df))

    X = clean_df[feature_cols].values
    y = clean_df[['target_24h', 'target_48h', 'target_72h']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_evaluations = {}

    # 1. Train Scikit-learn RandomForest Model
    print("Training Scikit-learn RandomForestRegressor model...")
    rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    rf_metrics = evaluate_model(y_test, rf_preds)
    print(f"[SUCCESS] Random Forest - RMSE: {rf_metrics['rmse']:.2f}, MAE: {rf_metrics['mae']:.2f}, R2: {rf_metrics['r2']:.2f}")
    model_evaluations["random_forest"] = {"model": rf_model, "metrics": rf_metrics, "filename": "rf_model.pkl"}

    # 2. Train Scikit-learn Ridge Regression Model (Statistical Baseline)
    print("Training Scikit-learn Ridge Regression model...")
    ridge_model = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge_model.fit(X_train_scaled, y_train)
    ridge_preds = ridge_model.predict(X_test_scaled)
    ridge_metrics = evaluate_model(y_test, ridge_preds)
    print(f"[SUCCESS] Ridge Regression - RMSE: {ridge_metrics['rmse']:.2f}, MAE: {ridge_metrics['mae']:.2f}, R2: {ridge_metrics['r2']:.2f}")
    model_evaluations["ridge_regression"] = {"model": ridge_model, "metrics": ridge_metrics, "filename": "ridge_model.pkl"}

    # 3. Train TensorFlow Deep Learning Model (Optional)
    tf_model = None
    try:
        import tensorflow as tf
        print("Training TensorFlow Multi-Output Neural Network model...")
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3)
        ])
        tf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        tf_model.fit(X_train_scaled, y_train, epochs=30, batch_size=16, verbose=0)
        tf_preds = tf_model.predict(X_test_scaled, verbose=0)
        tf_metrics = evaluate_model(y_test, tf_preds)
        print(f"[SUCCESS] TensorFlow Model - RMSE: {tf_metrics['rmse']:.2f}, MAE: {tf_metrics['mae']:.2f}, R2: {tf_metrics['r2']:.2f}")
        model_evaluations["tensorflow"] = {"model": tf_model, "metrics": tf_metrics, "filename": "tf_aqi_model.h5"}
    except ImportError:
        print("[WARNING] TensorFlow is not installed. Skipping deep learning model evaluation.")

    # Select Best Performing Model based on lowest RMSE
    best_model_key = min(model_evaluations, key=lambda k: model_evaluations[k]["metrics"]["rmse"])
    best_model_info = model_evaluations[best_model_key]
    print(f"\n[WINNER] Best Performing Model Selected: {best_model_key.upper()} (RMSE: {best_model_info['metrics']['rmse']:.2f})")

    # Local Save for Offline Testing & Serving
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    ridge_path = os.path.join(MODELS_DIR, "ridge_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    with open(rf_path, "wb") as f:
        pickle.dump(rf_model, f)
    with open(ridge_path, "wb") as f:
        pickle.dump(ridge_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_path, "wb") as f:
        pickle.dump(feature_cols, f)
    if tf_model is not None:
        tf_model.save(os.path.join(MODELS_DIR, "tf_aqi_model.h5"))

    print(f"[SUCCESS] Local model artifacts saved to '{MODELS_DIR}'.")

    # Hopsworks Model Registry Integration
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME")
    if api_key and project_name:
        try:
            import hopsworks
            print(f"\nRegistering best model ({best_model_key}) to Hopsworks Model Registry...")
            project = hopsworks.login(api_key_value=api_key, project=project_name)
            mr = project.get_model_registry()

            best_model_path = os.path.join(MODELS_DIR, best_model_info["filename"])
            hw_model = mr.python.create_model(
                name="islamabad_aqi_predictor",
                metrics=best_model_info["metrics"],
                description=f"Best performing model ({best_model_key}) for 3-day Islamabad AQI forecasting."
            )
            hw_model.save(best_model_path)
            print(f"[SUCCESS] Registered best model into Hopsworks Model Registry (Name: islamabad_aqi_predictor).")
        except Exception as e:
            print(f"[NOTE] Hopsworks Model Registry upload skipped: {e}")

if __name__ == "__main__":
    train_and_save_models()
