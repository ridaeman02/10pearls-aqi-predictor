import os
import sys
import datetime
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.feature_store import get_feature_store

# Load environment variables
load_dotenv()

def fetch_live_aqi_data(city: str) -> dict:
    """
    Fetches live AQI and weather data from AQICN (WAQI) API.
    If no token is provided, falls back to generating realistic mock live data.
    """
    aqicn_token = os.getenv("AQICN_TOKEN")
    
    if not aqicn_token:
        print("AQICN_TOKEN not found in environment. Generating simulated live data for testing...")
        now = datetime.datetime.now()
        np.random.seed(now.microsecond)
        pm25 = float(20 + 10 * np.sin(2 * np.pi * now.hour / 24) + np.random.normal(0, 5))
        pm10 = float(pm25 * 1.5 + np.random.normal(0, 3))
        no2 = float(10 + 5 * np.sin(2 * np.pi * now.hour / 24) + np.random.normal(0, 2))
        
        temp = float(25 + 5 * np.sin(2 * np.pi * now.hour / 24) + np.random.normal(0, 1))
        humidity = float(60 - 15 * np.sin(2 * np.pi * now.hour / 24) + np.random.normal(0, 5))
        wind_speed = float(abs(np.random.normal(4, 2)))
        pressure = float(1012.0 + np.random.normal(0, 2))
        
        aqi = int(0.5 * pm25 + 0.3 * pm10 + 0.2 * no2 + np.random.normal(0, 1))
        aqi = max(0, aqi)
        
        return {
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "temp": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "pressure": pressure,
            "aqi": aqi
        }

    # If AQICN Token is present, query the real API
    url = f"https://api.waqi.info/feed/{city}/?token={aqicn_token}"
    response = requests.get(url).json()
    
    if response.get("status") != "ok":
        raise Exception(f"Failed to fetch data from AQICN: {response.get('data')}")
        
    data = response["data"]
    iaqi = data.get("iaqi", {})
    
    return {
        "pm25": float(iaqi.get("pm25", {}).get("v", 25.0)),
        "pm10": float(iaqi.get("pm10", {}).get("v", 35.0)),
        "no2": float(iaqi.get("no2", {}).get("v", 15.0)),
        "temp": float(iaqi.get("t", {}).get("v", 25.0)),
        "humidity": float(iaqi.get("h", {}).get("v", 60.0)),
        "wind_speed": float(iaqi.get("w", {}).get("v", 5.0)),
        "pressure": float(iaqi.get("p", {}).get("v", 1013.0)),
        "aqi": int(data.get("aqi", 50))
    }

def get_latest_aqi_from_store(fs, city: str) -> float:
    """
    Reads the latest registered AQI value for the city from the feature store
    to compute the sequential change rate.
    """
    try:
        aqi_fg = fs.get_feature_group(name="aqi_predictions_fg", version=1)
        df = aqi_fg.read()
        if df is not None and not df.empty:
            city_df = df[df['city'] == city]
            if not city_df.empty:
                city_df_sorted = city_df.sort_values(by='timestamp', ascending=False)
                return float(city_df_sorted.iloc[0]['aqi'])
    except Exception as e:
        print(f"Could not retrieve latest AQI from feature store: {e}. Defaulting to baseline.")
    
    return 50.0 # Default fallback baseline AQI

def run_feature_pipeline():
    city = "Islamabad"
    
    # 1. Connect to Local Feature Store
    print("Connecting to Feature Store...")
    fs = get_feature_store()
    
    # 2. Fetch live measurements
    live_data = fetch_live_aqi_data(city)
    
    # 3. Compute time-based elements
    now = datetime.datetime.now()
    
    # 4. Compute derived metric: AQI Change Rate
    latest_aqi = get_latest_aqi_from_store(fs, city)
    current_aqi = live_data["aqi"]
    aqi_change_rate = float((current_aqi - latest_aqi) / (latest_aqi + 1e-5))
    
    # Structure feature record
    record = {
        'timestamp': [now.strftime('%Y-%m-%d %H:%M:%S')],
        'city': [city],
        'pm25': [live_data["pm25"]],
        'pm10': [live_data["pm10"]],
        'no2': [live_data["no2"]],
        'temp': [live_data["temp"]],
        'humidity': [live_data["humidity"]],
        'wind_speed': [live_data["wind_speed"]],
        'pressure': [live_data["pressure"]],
        'hour': [int(now.hour)],
        'day_of_week': [int(now.weekday())],
        'month': [int(now.month)],
        'aqi': [int(current_aqi)],
        'aqi_change_rate': [aqi_change_rate]
    }
    
    df = pd.DataFrame(record)
    print(f"New feature record to insert:\n{df.to_string(index=False)}")
    
    # 5. Insert record into Feature Group
    aqi_fg = fs.get_feature_group(name="aqi_predictions_fg", version=1)
    aqi_fg.insert(df)
    print("[SUCCESS] Successfully pushed live feature record to Feature Store.")

if __name__ == "__main__":
    run_feature_pipeline()
