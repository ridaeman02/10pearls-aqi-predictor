# Pearls AQI Predictor

A 100% serverless, end-to-end machine learning pipeline that forecasts the Air Quality Index (AQI) for the next 3 days.

## Project Structure

```text
pearls-aqi-predictor/
├── .github/
│   └── workflows/              # GitHub Actions for automated schedules (feature extraction, inference, deployment)
├── data_pipelines/             # Python scripts for serverless pipelines (run via GitHub Actions or locally)
│   ├── feature_pipeline.py     # Pulls hourly/daily weather & AQI data, updates Hopsworks Feature Store
│   ├── training_pipeline.py    # Fetches features, trains the model, and registers it to Model Registry
│   └── inference_pipeline.py   # Fetches newest data, predicts 3-day AQI, saves batch predictions
├── backfilling/                # Scripts to load historical AQI/weather data into Hopsworks
│   └── backfill.py
├── notebooks/                  # Jupyter Notebooks for research, analysis, and visualization
│   ├── eda.ipynb               # Exploratory Data Analysis
│   └── shap_analysis.ipynb     # Model interpretability with SHAP
├── dashboard/                  # Streamlit web application dashboard code
│   └── app.py                  # Streamlit entry point
├── .gitignore                  # Standard Python, environment, and credential file ignores
├── requirements.txt            # Project python dependencies
└── README.md                   # Project documentation
```

## Get Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
