import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "feature_store.db")

class LocalFeatureGroup:
    """
    Local Feature Group backed by SQLite & Pandas.
    """
    def __init__(self, name: str, db_path: str = DEFAULT_DB_PATH):
        self.name = name
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def insert(self, df: pd.DataFrame, write_options: dict = None):
        with sqlite3.connect(self.db_path) as conn:
            table_exists = pd.read_sql_query(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.name}'", conn
            )
            if not table_exists.empty:
                existing_df = pd.read_sql_query(f"SELECT * FROM {self.name}", conn)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                if 'city' in combined_df.columns and 'timestamp' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['city', 'timestamp'], keep='last')
                combined_df.to_sql(self.name, conn, if_exists="replace", index=False)
            else:
                df.to_sql(self.name, conn, if_exists="replace", index=False)
                
        print(f"[SUCCESS] [{self.name}] Saved {len(df)} record(s) into Feature Store.")

    def read(self) -> pd.DataFrame:
        if not os.path.exists(self.db_path):
            return pd.DataFrame()
        with sqlite3.connect(self.db_path) as conn:
            try:
                return pd.read_sql_query(f"SELECT * FROM {self.name}", conn)
            except Exception:
                return pd.DataFrame()

class DualFeatureStore:
    """
    Hybrid Feature Store router: attempts Hopsworks if API Key & Project are configured,
    otherwise falls back to Local SQLite Feature Store seamlessly.
    """
    def __init__(self):
        self.hopsworks_fs = None
        self.local_fs = LocalFeatureStore()
        
        api_key = os.getenv("HOPSWORKS_API_KEY")
        project_name = os.getenv("HOPSWORKS_PROJECT_NAME")
        
        if api_key and project_name:
            try:
                import hopsworks
                print(f"Connecting to Hopsworks Feature Store (Project: {project_name})...")
                project = hopsworks.login(api_key_value=api_key, project=project_name)
                self.hopsworks_fs = project.get_feature_store()
                print("[SUCCESS] Connected to Hopsworks Cloud Feature Store!")
            except Exception as e:
                print(f"[NOTE] Hopsworks Cloud connection skipped ({e}). Using Local Feature Store.")

    def get_or_create_feature_group(self, name: str, version: int = 1, primary_key: list = None, description: str = "", **kwargs):
        if self.hopsworks_fs:
            try:
                return self.hopsworks_fs.get_or_create_feature_group(
                    name=name,
                    version=version,
                    primary_key=primary_key or ['city', 'timestamp'],
                    description=description,
                    online_enabled=True
                )
            except Exception as e:
                print(f"[NOTE] Falling back to Local Feature Group ({e})")
        return self.local_fs.get_or_create_feature_group(name)

    def get_feature_group(self, name: str, version: int = 1, **kwargs):
        if self.hopsworks_fs:
            try:
                return self.hopsworks_fs.get_feature_group(name=name, version=version)
            except Exception as e:
                print(f"[NOTE] Falling back to Local Feature Group ({e})")
        return self.local_fs.get_feature_group(name)

class LocalFeatureStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path

    def get_or_create_feature_group(self, name: str, **kwargs) -> LocalFeatureGroup:
        return LocalFeatureGroup(name, self.db_path)

    def get_feature_group(self, name: str, **kwargs) -> LocalFeatureGroup:
        return LocalFeatureGroup(name, self.db_path)

def get_feature_store():
    return DualFeatureStore()
