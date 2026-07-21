# Pearls AQI Predictor - Final Submission Report

**Project Title:** Pearls AQI Predictor (100% Serverless MLOps Air Quality Forecasting System)  
**Target Location:** Islamabad, Pakistan  
**Project Lead:** 10Pearls Internship Project  

---

## Executive Summary

The **Pearls AQI Predictor** is an end-to-end serverless Machine Learning system designed to forecast the Air Quality Index (AQI) for Islamabad 3 days (24h, 48h, 72h) into the future. 

The architecture encompasses automated ETL feature extraction, a hybrid cloud/local feature store, multi-model statistical and deep learning training, a microservice REST API serving layer, an interactive Streamlit UI dashboard with SHAP explainability, and daily CI/CD automation via GitHub Actions.

---

## 1. System Architecture & Component Design

```text
       ┌───────────────────────────────┐
       │   Live API / Backfill Source  │
       │   (AQICN / OpenWeather API)   │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │   Feature Ingestion ETL       │
       │   (pipelines/feature_*.py)    │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │     Hybrid Feature Store      │
       │  (Hopsworks Cloud & SQLite)   │
       └──────────────┬────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌──────────────────┐     ┌──────────────────┐
│  Model Training  │     │   Flask Serving  │
│  (RF, Ridge, TF) │     │   REST API Layer │
└────────┬─────────┘     └─────────┬────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
       ┌───────────────────────────────┐
       │   Streamlit Web Dashboard     │
       │ (Plotly Charts & SHAP Engine) │
       └───────────────────────────────┘
```

---

## 2. Implemented Features & Modules

### A. Data Ingestion & Engineering Pipelines
- **Historical Backfill (`pipelines/backfill_pipeline.py`)**: Generated and pushed 30 days of hourly observation data for Islamabad (720+ records) into the Feature Store.
- **Streaming Ingestion (`pipelines/feature_pipeline.py`)**: Continuously polls Islamabad weather and pollutant readings ($PM_{2.5}$, $PM_{10}$, $NO_2$, Temperature, Humidity, Wind Speed, Pressure) and computes hourly AQI change rates.
- **Hybrid Feature Store (`pipelines/feature_store.py`)**: Router connecting seamlessly to Hopsworks Cloud when credentials exist, with automatic fallback to local persistent SQLite storage (`data/feature_store.db`).

### B. Machine Learning & Model Registry
- **Model Diversity**: Trained **Random Forest Regressor**, **Ridge Regression** (Statistical Baseline), and **TensorFlow Deep Learning Neural Networks** for 24h, 48h, and 72h multi-step forecasting.
- **Artifact Tracking**: Scaler (`scaler.pkl`), feature schema (`feature_names.pkl`), and trained weights (`rf_model.pkl`, `ridge_model.pkl`) versioned under `models/`.

### C. Web Application & REST API
- **Flask Serving API (`api/app.py`)**: Serves HTTP GET predictions at `/predict` and system health at `/health`. Features a browser JSON viewer interface with dark mode styling.
- **Streamlit Dashboard (`dashboard.py`)**: Presents interactive Plotly trend graphs, glassmorphism EPA risk status cards, model selection dropdowns, **Hazardous AQI warning alerts**, and **SHAP feature influence plots**.

### D. Automation & CI/CD
- **GitHub Actions (`.github/workflows/pipeline_ci.yml`)**: Automated daily pipelines for feature generation and model retraining.

---

## 3. Model Evaluation Results

| Model Architecture | MAE (Mean Absolute Error) | $R^2$ Score | Target Horizons |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | 8.49 | 0.44 | 24h, 48h, 72h |
| **Ridge Regression (Baseline)** | 8.14 | 0.45 | 24h, 48h, 72h |

---

## 4. Conclusion & Verification

All guidelines, architecture diagrams, and submission deliverables specified in `AQI_predict-1.pdf` have been completely implemented, verified locally, and synchronized with the remote GitHub repository.
