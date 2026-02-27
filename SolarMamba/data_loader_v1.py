import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import pvlib
from datetime import datetime, timedelta

class SolarDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform
        
        # Determine Root Path
        if config['env'] == 'server':
            self.root_dir = config['data']['server_root']
        else:
            self.root_dir = config['data']['local_root']
            
        self.months = config['data']['months']
        self.sequence_length = config['data']['sequence_length']
        self.sampling_rate = config['data']['sampling_rate_sec']
        self.horizons = config['model']['horizons'] # [1, 5, 10, 15]
        
        # 1. Load and Concatenate Data
        self.df = self._load_all_data()
        
        # 2. Feature Engineering (Physics-Informed)
        self.lat = 37.0916
        self.lon = -2.3636
        self.alt = 490.6
        self.df = self._add_solar_physics(self.df, self.lat, self.lon, self.alt)
        
        # 3. Filter Night Time (SZA > 85)
        self.df = self.df[self.df['SZA'] <= 85]
        
        # 4. Normalize Features
        feature_cols = ['k_index', 'temperature', 'pressure', 'SZA', 'Azimuth', 'sin_hour', 'cos_hour']
        self.feature_cols = feature_cols
        
        self.mean = self.df[feature_cols].mean()
        self.std = self.df[feature_cols].std()
        
        # Normalize k_index, Temp, Pressure, SZA, Azimuth
        # FIX: Added 'k_index' to normalization to ensure inputs and targets are in the same scale space
        cols_to_norm = ['k_index', 'temperature', 'pressure', 'SZA', 'Azimuth']
        self.df[cols_to_norm] = (self.df[cols_to_norm] - self.mean[cols_to_norm]) / (self.std[cols_to_norm] + 1e-6)
        
        # 5. Match Images
        self.samples = self._match_images()

    def _load_all_data(self):
        dfs = []
        for month in self.months:
            csv_path = os.path.join(self.root_dir, 'datasets', '1_Folsom', 'csv_files', f'Folsom_irradiance_weather.csv')
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found.")
                continue
            
            # Format: Date,GHI,DNI,DHI,temperature,pressure
            df = pd.read_csv(csv_path)
            
            # Parse Date: YYYYMMDDHHMMSS
            df['Datetime'] = pd.to_datetime(df['Date'], format='%Y%m%d%H%M%S')
            df['Month_Code'] = month # Keep track of source folder
            dfs.append(df)
            
        if not dfs:
            raise RuntimeError("No data files found!")
            
        master_df = pd.concat(dfs).sort_values('Datetime').set_index('Datetime')
        
        # Ensure numeric
        cols = ['GHI', 'DNI', 'DHI', 'temperature', 'pressure']
        for c in cols:
            master_df[c] = pd.to_numeric(master_df[c], errors='coerce')
            
        return master_df.dropna()

    def _add_solar_physics(self, df, lat, lon, alt):
        # Calculate Solar Position
        site = pvlib.location.Location(lat, lon, altitude=alt)
        solar_position = site.get_solarposition(df.index)
        
        df['SZA'] = solar_position['zenith']
        df['Azimuth'] = solar_position['azimuth']
        
        # Calculate Clear Sky GHI (Ineichen model)
        clearsky = site.get_clearsky(df.index, model='ineichen')
        df['GHI_cs'] = clearsky['ghi']
        
        # Calculate k* (Clear Sky Index)
        # Avoid division by zero
        df['k_index'] = df['GHI'] / (df['GHI_cs'] + 1e-6)
        # Clamp between 0.0 and 1.2
        df['k_index'] = df['k_index'].clip(0.0, 1.2)
        
        # Time features
        df['hour'] = df.index.hour + df.index.minute / 60.0
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        
        return df

    def _match_images(self):
        samples = []
        dataset_type = self.config['data'].get('dataset_type', 'undistorted')
        # We iterate through the dataframe to find valid samples
        # A valid sample has:
        # 1. Image file existing
        # 2. History (sequence_length * sampling_rate)
        # 3. Future Targets (for all horizons)
        
        # Pre-check image existence to avoid disk I/O in loop? 
        # No, structure is partitioned by month.
        
        valid_indices = []
        
        # Convert index to series for faster lookup
        timestamps = pd.Series(self.df.index, index=self.df.index)
        
        for dt in self.df.index:
            # 1. Check History
            start_dt = dt - timedelta(seconds=self.sampling_rate * (self.sequence_length - 1))
            if start_dt not in self.df.index:
                continue
                
            # 2. Check Targets
            has_targets = True
            for h in self.horizons:
                target_dt = dt + timedelta(minutes=h)
                if target_dt not in self.df.index:
                    has_targets = False
                    break
            if not has_targets:
                continue
                
            # 3. Check Image
            month_code = self.df.loc[dt, 'Month_Code']
            # Handle case where Month_Code might be a Series if duplicate indices exist (shouldn't happen with set_index but safe to check)
            if isinstance(month_code, pd.Series):
                month_code = month_code.iloc[0]
            
             # --- PATH SELECTION LOGIC ---
            if dataset_type == 'undistorted':
                # Structure: datasets/undistorted/09_Metas/DD/HH/YYYYMMDDHHmmss_00160_corrected.jpg
                day = dt.strftime("%d")
                hour = dt.strftime("%H")
                img_name = dt.strftime("%Y%m%d%H%M%S") + "_00160_corrected.jpg"
                img_path = os.path.join(self.root_dir, 'datasets', 'undistorted', f"{month_code}_Metas", day, hour, img_name)
            else:
                # Structure: datasets/original/metas_09/YYYYMMDDHHmmss.jpg
                # Note: Folder is 'metas_09', not '09_Metas'
                img_name = dt.strftime("%Y%m%d%H%M%S") + ".jpg"
                img_path = os.path.join(self.root_dir, 'datasets', 'original', f"metas_{month_code}", img_name)
            # ----------------------------
            
            if os.path.exists(img_path):
                samples.append((img_path, dt))
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, dt = self.samples[idx]
        
        # Load Image
        image = Image.open(img_path).convert('RGB')
        
        # Resize to 512x512 (as per prompt)
        image = image.resize((512, 512))
        
        # Masking
        mask = Image.new('L', (512, 512), 0)
        draw = ImageDraw.Draw(mask)
        center = (256, 256)
        radius = 250
        draw.ellipse((center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius), fill=255)
        image_np = np.array(image)
        mask_np = np.array(mask)
        image_np[mask_np == 0] = 0
        image = Image.fromarray(image_np)
        
        if self.transform:
            image = self.transform(image)
        
        # Get Weather Sequence
        # History: 60 steps back (30 mins)
        start_dt = dt - timedelta(seconds=self.sampling_rate * (self.sequence_length - 1))
        
        # We need to slice carefully. Since we filtered DF, rows might not be contiguous if there were gaps.
        # But we checked existence in _match_images.
        # Ideally we select by range, but we need exact steps.
        # Let's generate the expected timestamps
        seq_timestamps = [dt - timedelta(seconds=i*self.sampling_rate) for i in range(self.sequence_length)]
        seq_timestamps.reverse() # Oldest to newest
        
        # Select
        # Get weather sequence
        # Original code likely looks like: weather_slice = self.df.loc[seq_timestamps]
        
        # REPLACEMENT CODE:
        # Use reindex to handle missing timestamps by filling with nearest valid data or forward filling
        # method='nearest' finds the closest timestamp if exact match is missing
        # limit=1 ensures we don't grab data from too far away
        # FIX: Select only feature columns to ensure shape is (T, 7)
        weather_slice = self.df[self.feature_cols].reindex(seq_timestamps, method='nearest', tolerance=pd.Timedelta('10min'))
        
        # If reindex still leaves NaNs (because data is too far away), fill them
        if weather_slice.isnull().values.any():
            weather_slice = weather_slice.fillna(method='ffill').fillna(method='bfill').fillna(0.0)

        # Ensure data is numeric (fix for numpy.object_ error)
        # This forces conversion of any accidental strings to NaN, then fills them
        weather_values = weather_slice.values
        if weather_values.dtype == 'object':
            try:
                weather_values = weather_values.astype(float)
            except ValueError:
                # If direct conversion fails, use pandas to coerce errors
                weather_slice = weather_slice.apply(pd.to_numeric, errors='coerce').fillna(0.0)
                weather_values = weather_slice.values

        weather_seq = torch.tensor(weather_values, dtype=torch.float32)
        
        # Targets (Multi-Horizon)
        targets = []
        ghi_cs_targets = []
        
        for h in self.horizons:
            target_dt = dt + timedelta(minutes=h)
            k_val = self.df.loc[target_dt, 'k_index']
            ghi_cs_val = self.df.loc[target_dt, 'GHI_cs']
            
            # Handle potential duplicate index
            if isinstance(k_val, pd.Series): k_val = k_val.iloc[0]
            if isinstance(ghi_cs_val, pd.Series): ghi_cs_val = ghi_cs_val.iloc[0]
            
            targets.append(k_val)
            ghi_cs_targets.append(ghi_cs_val)
            
        targets = torch.tensor(targets, dtype=torch.float32) # Shape: [4]
        ghi_cs_targets = torch.tensor(ghi_cs_targets, dtype=torch.float32) # Shape: [4]
        
        return image, weather_seq, targets, ghi_cs_targets

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SolarDataset(config, transform=transform)
    
    # Chronological Split (80/20)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    
    # Indices are already chronological because samples were appended in chronological order of DF
    indices = list(range(total_len))
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    return train_loader, val_loader

