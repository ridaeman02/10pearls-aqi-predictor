# 🌍 Islamabad AQI Predictor

**Enterprise-Grade Air Quality Prediction System | 10Pearls Internship Project**

An end-to-end serverless MLOps pipeline that forecasts the Air Quality Index (AQI) for Islamabad, Pakistan for the next 3 days using historical weather metrics, pollutant features ($PM_{2.5}$, $PM_{10}$, $NO_2$), advanced feature engineering, and a multi-model architecture (Random Forest, Ridge Regression, TensorFlow Deep Learning).

**Key Highlights:**
- ✅ Automated daily training & feature ingestion pipeline via GitHub Actions CI/CD
- ✅ Cloud & Local Hybrid Feature Store (Hopsworks Cloud with seamless SQLite local fallback)
- ✅ Multi-model architecture (Scikit-learn & TensorFlow Neural Networks)
- ✅ Real-time interactive Streamlit web dashboard with 3-day Plotly interactive forecasts
- ✅ Microservice Flask REST API with automated browser JSON response viewer
- ✅ SHAP (SHapley Additive exPlanations) model interpretability
- ✅ Production-ready Python 3.10+ codebase

---

## 🚀 Key Features

*   **Multi-Model Architecture:** Trains and evaluates **Random Forest**, **Ridge Regression**, and **TensorFlow Deep Learning** models to predict 24h, 48h, and 72h future AQI offsets.
*   **Automated Pipelines:**
    *   **Feature Pipeline:** Fetches live data from AQICN/WAQI API for Islamabad, cleans noise, engineers lag & change-rate features, and pushes to the Feature Store.
    *   **Training Pipeline:** Retrains models daily, evaluates performance metrics ($RMSE, MAE, R^2$), and exports binaries to the local/cloud Model Registry.
*   **Advanced Forecasting Logic:**
    *   Predicts 24h, 48h, and 72h AQI into the future.
    *   Computes hourly AQI change rates relative to prior time steps.
*   **Interactive Streamlit Dashboard:**
    *   Real-time Islamabad AQI monitoring with glassmorphism visual cards.
    *   Plotly interactive historical and projected forecast charts.
    *   **🚨 Hazardous AQI Alert System:** Automatic warning callouts when AQI exceeds safe thresholds (100 / 150+).
    *   **🧠 SHAP Feature Interpretability:** Explains feature impact rankings (temperature, humidity, $PM_{2.5}$) on forecasts.
*   **Flask REST API Serving Gateway:**
    *   Serves predictions at `/predict` and health metrics at `/health`.
    *   Includes a browser HTML landing page and automatic JSON viewer.

---

## 🛠️ Tech Stack

*   **Language:** Python 3.10+
*   **Data Processing:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn (RandomForestRegressor, Ridge), TensorFlow / Keras (Multi-Output Neural Network)
*   **Interpretability:** SHAP
*   **Visualization:** Streamlit, Plotly Express, Matplotlib
*   **Feature Store & Storage:** Hopsworks Cloud / Local SQLite DB (`data/feature_store.db`)
*   **API Layer:** Flask
*   **CI/CD & Version Control:** GitHub Actions, Git

---

## 📂 Project Structure

```text
pearls-aqi-predictor/
├── .github/
│   └── workflows/              # GitHub Actions for daily automated pipelines
├── api/
│   └── app.py                  # Flask REST API serving layer
├── data/                       # Local feature store database (.db)
├── models/                     # Cache for model artifacts (.pkl, .h5)
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis notebook
├── pipelines/
│   ├── backfill_pipeline.py    # Historical data backfill pipeline (30 days)
│   ├── feature_pipeline.py     # Real-time Islamabad streaming ETL pipeline
│   ├── feature_store.py        # Hybrid Hopsworks / SQLite Feature Store router
│   └── training_pipeline.py    # Model training & registry export pipeline
├── .env                        # API Keys & Configurations (Not committed)
├── dashboard.py                # Streamlit Web Dashboard application
├── requirements.txt            # Python dependencies
└── README.md                   # Project Documentation
```

---

## ⚙️ Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/ridaeman02/10pearls-aqi-predictor.git
    cd 10pearls-aqi-predictor
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the root directory (optional for Hopsworks Cloud):
    ```ini
    HOPSWORKS_API_KEY=your_hopsworks_api_key
    HOPSWORKS_PROJECT_NAME=pearls_aqi_predictor
    AQICN_TOKEN=your_aqicn_token
    CITY=Islamabad
    ```

---

## 🏃 Usage Guide

### 1. Run Feature Backfill Pipeline (ETL)
Generates 30 days of historical data for Islamabad and updates the Feature Store.
```bash
python pipelines/backfill_pipeline.py
```

### 2. Run Real-Time Ingestion Pipeline
Fetches current weather/pollutant readings for Islamabad and pushes new feature vectors to the Feature Store.
```bash
python pipelines/feature_pipeline.py
```

### 3. Train ML Models
Trains Scikit-learn Random Forest and TensorFlow models, evaluates metrics, and exports artifacts to `models/`.
```bash
python pipelines/training_pipeline.py
```

### 4. Launch Streamlit Web Dashboard
Starts the interactive visualization dashboard.
```bash
python -m streamlit run dashboard.py
```

### 5. Launch Flask REST API Server
Starts the model serving REST API on port `5000`.
```bash
python api/app.py
```
*Access health check:* `http://localhost:5000/health`  
*Access prediction forecast:* `http://localhost:5000/predict`

---

## 🔄 CI/CD & Automation

This project uses **GitHub Actions** for fully automated serverless MLOps workflows.

### GitHub Actions Workflows ([`.github/workflows/pipeline_ci.yml`](.github/workflows/pipeline_ci.yml))

| Workflow Task | Schedule | Purpose |
| :--- | :--- | :--- |
| **Feature Extraction** | Hourly | Fetch & engineer features from AQICN API for Islamabad |
| **Model Retraining** | Daily (`0 0 * * *`) | Retrain ML models, evaluate metrics, update model registry |

---

## 🎯 Requirements & Fulfillment Tracker

### ✅ Phase 1: Automated Data Ingestion & Engineering (COMPLETED)

| Requirement | Solution | Status |
| :--- | :--- | :---: |
| Fetch weather & pollution data for Islamabad | AQICN / OpenWeather API integration with historical backfill | ✅ Completed |
| Collect AQI metrics ($PM_{2.5}, PM_{10}, NO_2$) | Integrated API endpoints for pollutant metrics | ✅ Completed |
| Clean and format data for ML pipelines | NaN handling, type-casting, timezone normalization | ✅ Completed |
| Time-series feature engineering | Hour, day of week, month, and derived `aqi_change_rate` | ✅ Completed |

---

### ✅ Phase 2: Cloud & Local Feature Store Integration (COMPLETED)

| Requirement | Solution | Status |
| :--- | :--- | :---: |
| Establish centralized ML feature repository | Hybrid `DualFeatureStore` (Hopsworks Cloud & SQLite local fallback) | ✅ Completed |
| Prevent data silos and ensure reproducibility | All features versioned with timestamp primary keys | ✅ Completed |
| Network resilience | Automatic local fallback if cloud credentials are empty | ✅ Completed |

---

### ✅ Phase 3: Model Training & Registry (COMPLETED)

| Requirement | Solution | Status |
| :--- | :--- | :---: |
| Train multiple ML models for comparison | Scikit-learn (Random Forest, Ridge Regression) & TensorFlow Neural Networks | ✅ Completed |
| Prevent data leakage in time-series forecasting | Chronological train/test split, shift targets for 24h, 48h, 72h forecasts | ✅ Completed |
| Evaluate models with time-series metrics | $RMSE, MAE, R^2$ score tracking per model | ✅ Completed |
| Model Registry export | Artifacts (`rf_model.pkl`, `scaler.pkl`, `feature_names.pkl`, `tf_aqi_model.h5`) versioned in `models/` | ✅ Completed |

---

### ✅ Phase 4: Interactive Dashboard & API Serving (COMPLETED)

| Requirement | Solution | Status |
| :--- | :--- | :---: |
| Real-time AQI monitoring & 3-day forecast | Streamlit UI + Plotly interactive charts | ✅ Completed |
| Hazardous alert system | Automatic warnings when AQI exceeds safe levels (>100 / >150) | ✅ Completed |
| Model interpretability | SHAP value feature importance bar charts | ✅ Completed |
| Serving API | Flask REST API with browser JSON Viewer | ✅ Completed |

---

## 🤝 Contribution Guidelines

1. **Feature Branches:** `git checkout -b feature/your-feature`
2. **Testing:** Run `python pipelines/training_pipeline.py` to validate pipeline changes
3. **Commits:** Clear, concise messages (e.g., `feat: Add Ridge regression baseline model`)

---

## 📧 Contact & Submission Info

**Project Lead:** 10Pearls Internship Program  
**Target City:** Islamabad, Pakistan  
**Repository:** [github.com/ridaeman02/10pearls-aqi-predictor](https://github.com/ridaeman02/10pearls-aqi-predictor)

---

**Last Updated:** 2026  
**Python Version:** 3.10+  
**License:** MIT
