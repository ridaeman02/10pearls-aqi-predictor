import os
import sys
import datetime
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# Custom Styling (Dark Glassmorphism UI Theme)
st.markdown("""
<style>
    body {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94A3B8;
    }
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 8px;
        font-weight: 600;
    }
    .advisory-card {
        background: rgba(30, 41, 59, 0.7);
        border-left: 5px solid #38BDF8;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions - Official EPA Categorization & Action Guidelines
def get_risk_badge(aqi_val: float):
    val = round(aqi_val)
    if val <= 50:
        return "Good", "🟢", "#22C55E", "Air quality is satisfactory. Enjoy outdoor physical activities!"
    elif val <= 100:
        return "Moderate", "🟡", "#EAB308", "Air quality is acceptable. Unusually sensitive individuals should monitor exertion."
    elif val <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#F97316", "Members of sensitive groups (asthma, children, elderly) should reduce prolonged outdoor exertion."
    elif val <= 200:
        return "Unhealthy", "🔴", "#EF4444", "Everyone may experience health effects. Wear N95 masks outdoors, keep windows closed, and run indoor air purifiers."
    elif val <= 300:
        return "Very Unhealthy", "🟣", "#A855F7", "Health alert: risk of health effects for everyone. Avoid outdoor physical activity and remain indoors."
    else:
        return "Hazardous", "🟤", "#881337", "Health warning of emergency conditions! Avoid all physical activity outdoors and run indoor HEPA air purifiers."

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
st.sidebar.markdown("## ⚙️ Settings Panel")
st.sidebar.markdown("Configured for **Islamabad** serverless pipelines.")

selected_model_name = st.sidebar.selectbox(
    "Select Forecasting Engine:",
    ["Random Forest Regressor", "TensorFlow Deep Learning"]
)

if st.sidebar.button("🔄 Trigger Live Feature Ingestion"):
    with st.spinner("Executing serverless Islamabad real-time stream..."):
        try:
            from pipelines.feature_pipeline import run_feature_pipeline
            run_feature_pipeline()
            st.sidebar.success("New feature data successfully synced!")
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Execution failed: {e}")

# Header Layout
st.markdown('<div class="main-title">🌤️ Islamabad AQI Predictor Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enterprise MLOps forecasting application with EPA standards & SHAP explainers</div>', unsafe_allow_html=True)

# Load Data & Models
df = load_feature_store_data()
rf_model, tf_model, scaler, feature_names = load_model_artifacts()

if df.empty:
    st.warning("⚠️ Feature Store database not populated. Please run backfill script.")
    st.stop()

# Get latest Islamabad record
islamabad_df = df[df['city'].str.lower() == 'islamabad'] if 'city' in df.columns else df
if islamabad_df.empty:
    islamabad_df = df

latest_record = islamabad_df.iloc[-1]
current_aqi = float(latest_record.get('aqi', 50))
risk_label, risk_emoji, risk_color, health_advice = get_risk_badge(current_aqi)

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

# Hazardous AQI Alert Banners
max_val = max(current_aqi, preds_24h, preds_48h, preds_72h)
if max_val > 150:
    st.error(f"🚨 **Hazardous / Unhealthy AQI Alert**: Forecasted AQI peaks at {int(max_val)} (Unhealthy). Wear N95 masks, keep windows closed, and run indoor air purifiers.")
elif max_val > 100:
    st.warning(f"⚠️ **Moderate / Sensitive Groups Advisory**: Forecasted AQI peaks at {int(max_val)} (Unhealthy for Sensitive Groups). Sensitive individuals should reduce outdoor exertion.")

# Interactive Metrics Grid (Custom Glassmorphism HTML cards)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current AQI</div>
        <div class="metric-value" style="color: {risk_color};">{int(current_aqi)}</div>
        <div class="metric-delta" style="color: {risk_color};">{risk_emoji} {risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    chg_24 = preds_24h - current_aqi
    col_24 = "#22C55E" if chg_24 <= 0 else "#EF4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">24-Hour Forecast</div>
        <div class="metric-value" style="color: #38BDF8;">{int(preds_24h)}</div>
        <div class="metric-delta" style="color: {col_24};">{chg_24:+.1f} change</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    chg_48 = preds_48h - current_aqi
    col_48 = "#22C55E" if chg_48 <= 0 else "#EF4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">48-Hour Forecast</div>
        <div class="metric-value" style="color: #38BDF8;">{int(preds_48h)}</div>
        <div class="metric-delta" style="color: {col_48};">{chg_48:+.1f} change</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    chg_72 = preds_72h - current_aqi
    col_72 = "#22C55E" if chg_72 <= 0 else "#EF4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">72-Hour Forecast</div>
        <div class="metric-value" style="color: #38BDF8;">{int(preds_72h)}</div>
        <div class="metric-delta" style="color: {col_72};">{chg_72:+.1f} change</div>
    </div>
    """, unsafe_allow_html=True)

# Dynamic Actionable EPA Health Advisory Card
st.markdown(f"""
<div class="advisory-card" style="border-left-color: {risk_color};">
    <h3 style="margin: 0 0 8px 0; font-size: 1.15rem; color: {risk_color};">
        {risk_emoji} Official EPA Health Action Guidelines ({risk_label})
    </h3>
    <p style="margin: 0; color: #E2E8F0; font-size: 0.95rem; line-height: 1.5;">
        {health_advice}
    </p>
</div>
""", unsafe_allow_html=True)

# Plotly Interactive Trend Graph
st.subheader("📈 Interactive AQI Historical & Forecast Trend")

hist_data = islamabad_df.tail(48).copy()
hist_times = hist_data['timestamp'].tolist()
hist_values = hist_data['aqi'].tolist()

last_time = hist_times[-1] if hist_times else datetime.datetime.now()
future_times = [
    last_time + datetime.timedelta(hours=24),
    last_time + datetime.timedelta(hours=48),
    last_time + datetime.timedelta(hours=72)
]
future_values = [preds_24h, preds_48h, preds_72h]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hist_times,
    y=hist_values,
    name="Historical AQI",
    line=dict(color="#38BDF8", width=3.5),
    mode="lines+markers"
))

fig.add_trace(go.Scatter(
    x=[last_time] + future_times,
    y=[current_aqi] + future_values,
    name="Projected Forecast",
    line=dict(color="#F59E0B", width=3.5, dash="dash"),
    mode="lines+markers"
))

fig.update_layout(
    plot_bgcolor="#0F172A",
    paper_bgcolor="#0F172A",
    font_color="#E2E8F0",
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="AQI Index"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Interactive Plotly SHAP Feature Importance
st.subheader("🧠 SHAP Feature Importance Interpretability")

if rf_model is not None:
    try:
        import shap
        explainer = shap.TreeExplainer(rf_model.estimators_[0])
        shap_values = explainer.shap_values(X_scaled)
        
        impacts = np.abs(shap_values[0]) if len(shap_values.shape) > 1 else np.abs(shap_values)
        
        shap_df = pd.DataFrame({
            "Feature": [col.replace("_", " ").upper() for col in feature_cols],
            "Impact": impacts
        }).sort_values(by="Impact", ascending=True)

        fig_shap = px.bar(
            shap_df,
            x="Impact",
            y="Feature",
            orientation="h",
            color="Impact",
            color_continuous_scale="Purples",
            labels={"Impact": "Absolute Influence Score"}
        )

        fig_shap.update_layout(
            plot_bgcolor="#0F172A",
            paper_bgcolor="#0F172A",
            font_color="#E2E8F0",
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=False)
        )

        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.plotly_chart(fig_shap, use_container_width=True)
        with col_r:
            st.markdown("#### Impact Rankings")
            st.dataframe(
                shap_df.sort_values(by="Impact", ascending=False).style.background_gradient(cmap="Purples"),
                use_container_width=True
            )

    except Exception as e:
        if hasattr(rf_model, "estimators_"):
            importances = np.mean([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
            imp_df = pd.DataFrame({
                "Feature": [col.replace("_", " ").upper() for col in feature_cols],
                "Importance": importances
            }).sort_values(by="Importance", ascending=True)

            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color="Importance")
            fig_imp.update_layout(plot_bgcolor="#0F172A", paper_bgcolor="#0F172A", font_color="#E2E8F0")
            st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Train your model to display interactive feature explainability features.")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Pearls AQI Predictor App | Islamabad MLOps serving system")
