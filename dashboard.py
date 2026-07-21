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
    page_title="Islamabad AQI Predictor | MLOps Platform",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Controls & Theme Selector
st.sidebar.markdown("## ⚙️ Settings Panel")
st.sidebar.markdown("Target Region: **Islamabad, Pakistan**")

# Sun / Moon Theme Switcher
theme_mode = st.sidebar.radio(
    "🎨 Display Theme Mode:",
    ["🌙 Dark Mode", "☀️ Light Mode"],
    index=0
)
is_dark = "Dark" in theme_mode

selected_model_name = st.sidebar.selectbox(
    "Active Forecasting Engine:",
    ["Ridge Regression (Statistical)", "Random Forest Regressor", "TensorFlow Deep Learning"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Serverless ETL")
if st.sidebar.button("Sync Live Ingestion Stream"):
    with st.spinner("Fetching Islamabad live weather stream..."):
        try:
            from pipelines.feature_pipeline import run_feature_pipeline
            run_feature_pipeline()
            st.sidebar.success("Stream synchronized!")
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Sync failed: {e}")

# Define Theme Variables
if is_dark:
    bg_color = "#0B0F19"
    card_bg = "rgba(22, 30, 49, 0.65)"
    border_color = "rgba(255, 255, 255, 0.08)"
    text_color = "#F8FAFC"
    muted_text = "#94A3B8"
    plotly_bg = "#0F172A"
    grid_color = "rgba(255, 255, 255, 0.05)"
    shadow = "0 12px 30px rgba(0, 0, 0, 0.3)"
else:
    bg_color = "#F8FAFC"
    card_bg = "#FFFFFF"
    border_color = "rgba(0, 0, 0, 0.08)"
    text_color = "#0F172A"
    muted_text = "#475569"
    plotly_bg = "#FFFFFF"
    grid_color = "rgba(0, 0, 0, 0.06)"
    shadow = "0 12px 30px rgba(0, 0, 0, 0.05)"

# Apply Dynamic CSS Based on Selected Theme Mode
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    
    .hero-container {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 24px;
        padding: 30px 40px;
        margin-bottom: 25px;
        box-shadow: {shadow};
    }}
    
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10B981;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }}

    .pulse-dot {{
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
    }}

    .main-title {{
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0284C7 0%, #6366F1 50%, #9333EA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        letter-spacing: -1px;
    }}
    
    .subtitle {{
        font-size: 1.05rem;
        color: {muted_text};
        margin: 0;
        line-height: 1.6;
    }}

    .metric-grid-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: {shadow};
        transition: all 0.3s ease;
    }}

    .metric-grid-card:hover {{
        transform: translateY(-5px);
        border-color: #0284C7;
    }}

    .card-label {{
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: {muted_text};
        margin-bottom: 8px;
    }}

    .card-value {{
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1;
        margin: 10px 0;
    }}

    .card-tag {{
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 10px;
        margin-top: 6px;
    }}

    .epa-advisory-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-left-width: 6px;
        border-radius: 18px;
        padding: 24px;
        margin: 25px 0;
        box-shadow: {shadow};
    }}

    .epa-title {{
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .epa-desc {{
        color: {text_color};
        font-size: 0.98rem;
        line-height: 1.6;
        margin: 0;
    }}

    [data-testid="stSidebar"] {{
        background-color: {"#07090E" if is_dark else "#F1F5F9"};
        border-right: 1px solid {border_color};
    }}

    .stButton>button {{
        width: 100%;
        background: linear-gradient(135deg, #0284C7, #6366F1);
        color: #FFFFFF;
        border: none;
        border-radius: 12px;
        padding: 12px;
        font-weight: 700;
        font-size: 0.95rem;
    }}
</style>
""", unsafe_allow_html=True)

# Official EPA Categorization & Action Guidelines
def get_risk_badge(aqi_val: float):
    val = round(aqi_val)
    if val <= 50:
        return "Good", "🟢", "#22C55E", "Air quality is satisfactory. Enjoy outdoor physical activities!"
    elif val <= 100:
        return "Moderate", "🟡", "#D97706" if not is_dark else "#EAB308", "Air quality is acceptable. Unusually sensitive individuals should monitor exertion."
    elif val <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#EA580C" if not is_dark else "#F97316", "Members of sensitive groups (children, elderly) should reduce prolonged outdoor exertion."
    elif val <= 200:
        return "Unhealthy", "🔴", "#DC2626" if not is_dark else "#EF4444", "Everyone may experience health effects. Wear N95 masks outdoors, keep windows closed, and run indoor air purifiers."
    elif val <= 300:
        return "Very Unhealthy", "🟣", "#9333EA" if not is_dark else "#A855F7", "Health alert: risk of health effects for everyone. Avoid outdoor physical activity and remain indoors."
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
    ridge_path = os.path.join(models_dir, "ridge_model.pkl")
    tf_path = os.path.join(models_dir, "tf_aqi_model.h5")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")

    rf_model, ridge_model, tf_model, scaler, feature_names = None, None, None, None, None

    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
    if os.path.exists(ridge_path):
        with open(ridge_path, "rb") as f:
            ridge_model = pickle.load(f)
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

    return rf_model, ridge_model, tf_model, scaler, feature_names

# Main Hero Header with Theme Banner Icon
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="status-badge">
            <span class="pulse-dot"></span> Live MLOps Stream
        </div>
        <div style="font-size: 1.4rem;">
            {"🌙" if is_dark else "☀️"}
        </div>
    </div>
    <div class="main-title">Islamabad AQI Predictor</div>
    <div class="subtitle">Real-time air quality forecasting engine & 3-day multi-step predictions.</div>
</div>
""", unsafe_allow_html=True)

# Load Data & Models
df = load_feature_store_data()
rf_model, ridge_model, tf_model, scaler, feature_names = load_model_artifacts()

if df.empty:
    st.warning("⚠️ Feature Store is empty. Please run backfill script.")
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

if selected_model_name == "Ridge Regression (Statistical)" and ridge_model is not None:
    try:
        raw_preds = ridge_model.predict(X_scaled)[0]
        preds_24h, preds_48h, preds_72h = float(raw_preds[0]), float(raw_preds[1]), float(raw_preds[2])
    except Exception:
        pass
elif selected_model_name == "TensorFlow Deep Learning" and tf_model is not None:
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

# Metric Grid Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="card-label">Current AQI</div>
        <div class="card-value" style="color: {risk_color};">{int(current_aqi)}</div>
        <div class="card-tag" style="background: {risk_color}20; color: {risk_color};">{risk_emoji} {risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    chg_24 = preds_24h - current_aqi
    col_24 = "#10B981" if chg_24 <= 0 else "#EF4444"
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="card-label">24-Hour Forecast</div>
        <div class="card-value" style="color: #0284C7;">{int(preds_24h)}</div>
        <div class="card-tag" style="background: {col_24}20; color: {col_24};">{chg_24:+.1f} Shift</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    chg_48 = preds_48h - current_aqi
    col_48 = "#10B981" if chg_48 <= 0 else "#EF4444"
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="card-label">48-Hour Forecast</div>
        <div class="card-value" style="color: #6366F1;">{int(preds_48h)}</div>
        <div class="card-tag" style="background: {col_48}20; color: {col_48};">{chg_48:+.1f} Shift</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    chg_72 = preds_72h - current_aqi
    col_72 = "#10B981" if chg_72 <= 0 else "#EF4444"
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="card-label">72-Hour Forecast</div>
        <div class="card-value" style="color: #9333EA;">{int(preds_72h)}</div>
        <div class="card-tag" style="background: {col_72}20; color: {col_72};">{chg_72:+.1f} Shift</div>
    </div>
    """, unsafe_allow_html=True)

# Official EPA Action Guidelines Card
st.markdown(f"""
<div class="epa-advisory-card" style="border-left-color: {risk_color};">
    <div class="epa-title" style="color: {risk_color};">
        <span>{risk_emoji}</span> Official EPA Action Guidelines ({risk_label})
    </div>
    <p class="epa-desc">{health_advice}</p>
</div>
""", unsafe_allow_html=True)

# Plotly Interactive Trend Graph
st.markdown("### 📈 Historical AQI & 3-Day Forecast Trend")

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

# Historical Line
fig.add_trace(go.Scatter(
    x=hist_times,
    y=hist_values,
    name="Historical Observed AQI",
    fill='tozeroy',
    fillcolor='rgba(2, 132, 199, 0.1)',
    line=dict(color="#0284C7", width=3),
    mode="lines+markers",
    marker=dict(size=6, color="#0284C7")
))

# Forecast Line
fig.add_trace(go.Scatter(
    x=[last_time] + future_times,
    y=[current_aqi] + future_values,
    name="Projected ML Forecast",
    line=dict(color="#9333EA", width=3.5, dash="dash"),
    mode="lines+markers",
    marker=dict(size=8, color="#9333EA")
))

fig.update_layout(
    plot_bgcolor=plotly_bg,
    paper_bgcolor=plotly_bg,
    font_color=text_color,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(showgrid=True, gridcolor=grid_color),
    yaxis=dict(showgrid=True, gridcolor=grid_color, title="Air Quality Index (AQI)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# SHAP Feature Importance Section
st.markdown("### 🧠 SHAP Feature Importance Interpretability")

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
            color_continuous_scale="Purples" if is_dark else "Blues",
            labels={"Impact": "Absolute Influence Score"}
        )

        fig_shap.update_layout(
            plot_bgcolor=plotly_bg,
            paper_bgcolor=plotly_bg,
            font_color=text_color,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor=grid_color),
            yaxis=dict(showgrid=False)
        )

        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.plotly_chart(fig_shap, use_container_width=True)
        with col_r:
            st.markdown("#### Feature Rank Table")
            st.dataframe(
                shap_df.sort_values(by="Impact", ascending=False).style.background_gradient(cmap="Blues"),
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
            fig_imp.update_layout(plot_bgcolor=plotly_bg, paper_bgcolor=plotly_bg, font_color=text_color)
            st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Train your model to display interactive feature explainability features.")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Pearls AQI Predictor | 10Pearls Internship Project | Islamabad Serverless MLOps System")
