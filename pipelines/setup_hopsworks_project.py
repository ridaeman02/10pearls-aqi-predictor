import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

def setup_project(project_name: str = "pearls_aqi_predictor"):
    print("Connecting to Hopsworks...")
    try:
        conn = hopsworks.connection()
        projects = conn.get_projects()
        print(f"Found {len(projects)} existing project(s): {[p.name for p in projects]}")
        
        # Check if project already exists
        for p in projects:
            if p.name.lower() == project_name.lower():
                print(f"Project '{p.name}' already exists!")
                return p
        
        # Try to create project programmatically
        print(f"Attempting to create project '{project_name}' programmatically...")
        project = conn.create_project(
            name=project_name,
            description="Pearls AQI Predictor MLOps Project"
        )
        print(f"Successfully created project '{project.name}'!")
        return project

    except Exception as e:
        print(f"\nCould not list/create project automatically: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure HOPSWORKS_API_KEY in .env is valid and has admin/project creation privileges.")
        print("2. Check if your Hopsworks SaaS quota has reached maximum allowed projects.")
        return None

if __name__ == "__main__":
    setup_project()
