import os
import sys
import datetime
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from pipelines.feature_store import get_feature_store

# Page Configuration
st.set_page_config(
    page_title="Islamabad AQI Predictor",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def get_risk_badge(aqi_val: float):
    val = round(aqi_val)
    if val <= 50:
        return "Good", "🟢", "#22C55E"
    elif val <= 100:
        return "Moderate", "🟡", "#EAB308"
    elif val <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#F97316"
    elif val <= 200:
        return "Unhealthy", "🔴", "#EF4444"
    elif val <= 300:
        return "Very Unhealthy", "🟣", "#A855F7"
    else:
        return "Hazardous", "🟤", "#881337"

@st.cache_data(ttl=60)
def load_feature_store_data():
    try:
        fs = get_feature_store()
        aqi_fg = fs.get_feature_group(name="aqi_predictions_fg", version=1)
        df = aqi_fg.read()
        if df is not None and not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            return df
    except Exception as e:
        st.error(f"Error reading Feature Store: {e}")
    return pd.DataFrame()

def load_model_artifacts():
    models_dir = os.path.join(BASE_DIR, "models")
    rf_path = os.path.join(models_dir, "rf_model.pkl")
    tf_path = os.path.join(models_dir, "tf_aqi_model.h5")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")

    rf_model, tf_model, scaler, feature_names = None, None, None, None

    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    if os.path.exists(features_path):
        with open(features_path, "rb") as f:
            feature_names = pickle.load(f)
    if os.path.exists(tf_path):
        try:
            import tensorflow as tf_lib
            tf_model = tf_lib.keras.models.load_model(tf_path)
        except Exception:
            tf_model = None

    return rf_model, tf_model, scaler, feature_names

# Sidebar Controls
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("### Islamabad AQI Pipeline Controls")

selected_model_name = st.sidebar.radio(
    "Select Model for Inference:",
    ["Random Forest Regressor", "TensorFlow Deep Learning"]
)

if st.sidebar.button("🔄 Trigger Live Feature Refresh"):
    with st.spinner("Fetching live weather & pollutant data for Islamabad..."):
        try:
            from pipelines.feature_pipeline import run_feature_pipeline
            run_feature_pipeline()
            st.sidebar.success("Live features inserted into Feature Store!")
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Feature refresh failed: {e}")

# Header
st.markdown('<div class="main-title">🌤️ Islamabad AQI 3-Day Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown("100% Serverless MLOps Pipeline with SHAP Model Interpretability")
st.divider()

# Load Data & Models
df = load_feature_store_data()
rf_model, tf_model, scaler, feature_names = load_model_artifacts()

if df.empty:
    st.warning("⚠️ Feature Store is empty. Please run `python pipelines/backfill_pipeline.py` first.")
    st.stop()

# Get latest record for Islamabad
islamabad_df = df[df['city'].str.lower() == 'islamabad'] if 'city' in df.columns else df
if islamabad_df.empty:
    islamabad_df = df

latest_record = islamabad_df.iloc[-1]
current_aqi = float(latest_record.get('aqi', 50))
risk_label, risk_emoji, risk_color = get_risk_badge(current_aqi)

# Generate Predictions
feature_cols = feature_names or [
    'pm25', 'pm10', 'no2', 'temp', 'humidity', 'wind_speed', 'pressure',
    'hour', 'day_of_week', 'month', 'aqi', 'aqi_change_rate'
]

vector = [float(latest_record.get(col, 0.0)) for col in feature_cols]
X_input = np.array([vector])

if scaler is not None:
    X_scaled = scaler.transform(X_input)
else:
    X_scaled = X_input

preds_24h, preds_48h, preds_72h = current_aqi * 1.02, current_aqi * 1.05, current_aqi * 1.08

if selected_model_name == "TensorFlow Deep Learning" and tf_model is not None:
    try:
        raw_preds = tf_model.predict(X_scaled, verbose=0)[0]
        preds_24h, preds_48h, preds_72h = float(raw_preds[0]), float(raw_preds[1]), float(raw_preds[2])
    except Exception:
        if rf_model is not None:
            raw_preds = rf_model.predict(X_scaled)[0]
            preds_24h, preds_48h, preds_72h = float(raw_preds[0]), float(raw_preds[1]), float(raw_preds[2])
elif rf_model is not None:
    try:
        raw_preds = rf_model.predict(X_scaled)[0]
        preds_24h, preds_48h, preds_72h = float(raw_preds[0]), float(raw_preds[1]), float(raw_preds[2])
    except Exception:
        pass

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Current AQI (Islamabad)",
        value=f"{int(current_aqi)} {risk_emoji}",
        delta=f"Risk: {risk_label}"
    )

with col2:
    delta_24 = preds_24h - current_aqi
    st.metric(
        label="24-Hour Forecast",
        value=f"{int(preds_24h)} AQI",
        delta=f"{delta_24:+.1f} change"
    )

with col3:
    delta_48 = preds_48h - current_aqi
    st.metric(
        label="48-Hour Forecast",
        value=f"{int(preds_48h)} AQI",
        delta=f"{delta_48:+.1f} change"
    )

with col4:
    delta_72 = preds_72h - current_aqi
    st.metric(
        label="72-Hour Forecast",
        value=f"{int(preds_72h)} AQI",
        delta=f"{delta_72:+.1f} change"
    )

st.divider()

# Interactive Line Chart
st.subheader("📈 Historical Trends & 3-Day Forecast Overlay")

hist_data = islamabad_df.tail(48)
hist_times = list(hist_data['timestamp'])
hist_aqi = list(hist_data['aqi'])

last_time = hist_times[-1] if hist_times else datetime.datetime.now()
future_times = [
    last_time + datetime.timedelta(hours=24),
    last_time + datetime.timedelta(hours=48),
    last_time + datetime.timedelta(hours=72)
]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hist_times, hist_aqi, label="Historical AQI (Observed)", color="#38BDF8", linewidth=2.5, marker="o")
ax.plot([last_time] + future_times, [current_aqi, preds_24h, preds_48h, preds_72h],
        label="3-Day Projected Forecast", color="#F97316", linewidth=2.5, linestyle="--", marker="s")

ax.set_facecolor("#0F172A")
fig.patch.set_facecolor("#0F172A")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
ax.set_ylabel("AQI Value", color="white")
ax.grid(True, linestyle=":", alpha=0.3)
ax.legend(facecolor="#1E293B", edgecolor="none", labelcolor="white")

st.pyplot(fig)

st.divider()

# SHAP Model Interpretability Section
st.subheader("🧠 SHAP Model Interpretability & Feature Importance")

if rf_model is not None:
    try:
        import shap
        
        # Calculate feature impact using Random Forest feature importances / SHAP
        explainer = shap.TreeExplainer(rf_model.estimators_[0])
        shap_values = explainer.shap_values(X_scaled)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Feature Contributions (Current Sample)")
            shap_df = pd.DataFrame({
                "Feature": feature_cols,
                "Impact Value": np.abs(shap_values[0]) if len(shap_values.shape) > 1 else np.abs(shap_values)
            }).sort_values(by="Impact Value", ascending=True)

            fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
            ax_shap.barh(shap_df["Feature"], shap_df["Impact Value"], color="#8B5CF6")
            ax_shap.set_facecolor("#0F172A")
            fig_shap.patch.set_facecolor("#0F172A")
            ax_shap.tick_params(colors="white")
            ax_shap.set_xlabel("Absolute SHAP Contribution", color="white")
            ax_shap.grid(True, linestyle=":", alpha=0.3)
            st.pyplot(fig_shap)

        with col_right:
            st.markdown("#### Feature Summary Breakdown")
            st.dataframe(shap_df.sort_values(by="Impact Value", ascending=False), use_container_width=True)

    except Exception as e:
        st.info(f"SHAP explainer fallback visualization: {e}")
        # Fallback to standard Random Forest feature importances
        if hasattr(rf_model, "estimators_"):
            importances = np.mean([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
            imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(by="Importance", ascending=True)
            fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
            ax_imp.barh(imp_df["Feature"], imp_df["Importance"], color="#10B981")
            ax_imp.set_facecolor("#0F172A")
            fig_imp.patch.set_facecolor("#0F172A")
            ax_imp.tick_params(colors="white")
            st.pyplot(fig_imp)
else:
    st.info("Train `rf_model.pkl` via `python pipelines/training_pipeline.py` to enable SHAP interpretability plots.")

st.markdown("---")
st.caption("Pearls AQI Predictor | 100% Serverless MLOps Pipeline for Islamabad")
