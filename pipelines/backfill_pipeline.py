import os
import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import hopsworks

# Load environment variables
load_dotenv()

def generate_historical_data(city: str, days: int = 365) -> pd.DataFrame:
    """
    Simulates historical weather and AQI data for backfilling.
    In production, this would fetch from historical APIs or databases.
    """
    print(f"Generating {days} days of historical weather and AQI data for {city}...")
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')  # Hourly data
    
    # Simulate features
    np.random.seed(42)
    n_samples = len(date_range)
    
    # 1. Weather features
    temp = 15 + 10 * np.sin(2 * np.pi * date_range.dayofyear / 365) + np.random.normal(0, 3, n_samples)
    humidity = 60 - 20 * np.sin(2 * np.pi * date_range.dayofyear / 365) + np.random.normal(0, 10, n_samples)
    humidity = np.clip(humidity, 10, 100)
    wind_speed = np.abs(np.random.normal(5, 3, n_samples))
    pressure = 1013.25 + np.random.normal(0, 5, n_samples)
    
    # 2. Raw Pollutant Metrics
    pm25 = np.maximum(5, 20 + 15 * np.sin(2 * np.pi * date_range.dayofyear / 365) + np.random.normal(0, 10, n_samples))
    pm10 = pm25 * 1.5 + np.random.normal(0, 5, n_samples)
    pm10 = np.maximum(5, pm10)
    no2 = np.maximum(2, 10 + 5 * np.sin(2 * np.pi * date_range.hour / 24) + np.random.normal(0, 4, n_samples))
    
    # Simulate current hourly AQI based on pollutants
    aqi = 0.5 * pm25 + 0.3 * pm10 + 0.2 * no2 + np.random.normal(0, 2, n_samples)
    aqi = np.maximum(0, aqi).astype(int)
    
    # 3. Derived Metrics (AQI Change Rate over last hour)
    aqi_shift = np.roll(aqi, 1)
    aqi_shift[0] = aqi[0]
    aqi_change_rate = (aqi - aqi_shift) / (aqi_shift + 1e-5)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range.strftime('%Y-%m-%d %H:%M:%S'),
        'city': city,
        # Pollutants
        'pm25': pm25.astype(float),
        'pm10': pm10.astype(float),
        'no2': no2.astype(float),
        # Weather
        'temp': temp.astype(float),
        'humidity': humidity.astype(float),
        'wind_speed': wind_speed.astype(float),
        'pressure': pressure.astype(float),
        # Time-based
        'hour': date_range.hour.astype(int),
        'day_of_week': date_range.dayofweek.astype(int),
        'month': date_range.month.astype(int),
        # Target & Derived
        'aqi': aqi.astype(int),
        'aqi_change_rate': aqi_change_rate.astype(float)
    })
    
    return df

def backfill_to_feature_store():
    # 1. Connect to Hopsworks Feature Store
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("Warning: HOPSWORKS_API_KEY environment variable not set. Will prompt in console if not logged in.")
        
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # Generate historical data for Karachi
    df = generate_historical_data(city="Karachi", days=30)
    
    # 2. Define schema & Create/Get Feature Group
    print("Creating/Getting Feature Group on Hopsworks...")
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_predictions_fg",
        version=1,
        primary_key=['city', 'timestamp'],
        description="AQI and Weather monitoring hourly feature group including raw pollutants and derived features.",
        online_enabled=True,
        statistics_config={"enabled": True, "histograms": True, "correlations": True}
    )
    
    # 3. Insert data into the Feature Group
    print("Inserting data into Hopsworks Feature Store...")
    # The insert method automatically creates the schema based on pandas dataframe types
    aqi_fg.insert(df, write_options={"wait_for_job": False})
    print("Data backfill successfully triggered in Hopsworks.")

if __name__ == "__main__":
    backfill_to_feature_store()
