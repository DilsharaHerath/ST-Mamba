import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import argparse
import yaml

def generate_mock_data(config_path='../config.yaml'):
    # Load config to get paths
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force local environment for mock generation
    root_dir = config['data']['local_root']
    months = config['data']['months']
    
    print(f"Generating mock data in {root_dir}...")
    
    # Create directory structure
    csv_dir = os.path.join(root_dir, 'datasets', 'full_dataset')
    img_root = os.path.join(root_dir, 'datasets', 'undistorted')
    
    os.makedirs(csv_dir, exist_ok=True)
    
    # Generate data for each month
    start_dates = {
        '09': datetime(2019, 9, 1, 6, 0, 0),
        '10': datetime(2019, 10, 1, 6, 0, 0),
        '11': datetime(2019, 11, 1, 6, 0, 0)
    }
    
    for month in months:
        print(f"Processing Month {month}...")
        
        # 1. Create CSV
        # Format: Date,GHI,DNI,DHI,temperature,pressure
        # Date format: YYYYMMDDHHMMSS
        
        # Generate 1 day of data per month for mock purposes (sampling every 30s)
        # 12 hours * 120 samples/hr = 1440 samples
        num_samples = 1440
        current_dt = start_dates[month]
        
        data = []
        img_folder = os.path.join(img_root, f"{month}_Metas")
        os.makedirs(img_folder, exist_ok=True)
        
        for _ in range(num_samples):
            dt_str = current_dt.strftime("%Y%m%d%H%M%S")
            
            # Mock values
            ghi = max(0, 1000 * np.sin(np.pi * (_ / num_samples))) # Simple curve
            dni = ghi * 0.8
            dhi = ghi * 0.2
            temp = 25.0
            press = 1013.0
            
            data.append([dt_str, ghi, dni, dhi, temp, press])
            
            # 2. Create Mock Image (Black image)
            # Only create a few images to save space/time, or all?
            # Let's create all to ensure data loader works, but small size
            # Prompt says 2048x2048, but for mock we can do small, loader resizes anyway.
            # Actually loader expects to resize, so input size doesn't strictly matter as long as it opens.
            img_path = os.path.join(img_folder, f"{dt_str}.png")
            if not os.path.exists(img_path):
                # Create a small placeholder image
                Image.new('RGB', (100, 100), color='black').save(img_path)
            
            current_dt += timedelta(seconds=30)
            
        df = pd.DataFrame(data, columns=['Date', 'GHI', 'DNI', 'DHI', 'temperature', 'pressure'])
        csv_path = os.path.join(csv_dir, f"metas_{month}_labels.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Created {csv_path} and {num_samples} images in {img_folder}")

    print("Mock data generation complete.")

if __name__ == "__main__":
    # Assume script is run from utils/
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    generate_mock_data(config_path)
