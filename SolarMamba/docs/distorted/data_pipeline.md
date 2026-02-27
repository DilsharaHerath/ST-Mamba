# Data Pipeline

## Overview
The data pipeline transforms raw images and meteorological data into a format suitable for the hybrid model.

## Steps

### 1. Metadata Parsing
- **Source:** `PSA_timeSeries_Metas.csv`
- **Coordinates:** Lat: 37.0916, Lon: -2.3636, Alt: 490.6
- **Index:** DatetimeIndex created from Date and Time columns.

### 2. Physics-Informed Feature Engineering
- **Library:** `pvlib`
- **Calculations:**
    - **Solar Zenith Angle (SZA)**
    - **Solar Azimuth Angle**
    - **Clear Sky GHI (GHI_cs):** Using the Ineichen model.
    - **Clear Sky Index (k*):** $k^* = GHI / GHI_{cs}$. Clamped to [0.0, 1.2].
    - **Time Features:** Sine and Cosine of the hour.

### 3. Filtering
- **Night Filter:** Data points with SZA > 85° are removed.

### 4. Image Processing
- **Resize:** 512x512 pixels.
- **Masking:** A circular mask (radius 250) is applied to remove the horizon and ground artifacts.
- **Normalization:** Standard ImageNet mean and std.

### 5. Sequence Generation
- **History:** 60 minutes of past weather data.
- **Features:** `[k_index, Temp, Pressure, SZA, Azimuth, sin_hour, cos_hour]`
- **Normalization:** Temp, Pressure, SZA, and Azimuth are normalized using standard scaling.
