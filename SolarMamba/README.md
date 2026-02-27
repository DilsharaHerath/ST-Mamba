# SolarMamba: Physics-Informed Mamba-Ladder for Solar Nowcasting

## Overview
SolarMamba is a research-grade solar irradiance forecasting model that integrates:
1.  **Visual Backbone:** Nvidia MambaVision (Base) for extracting spatial features from All-Sky Imager (ASI) images.
2.  **Temporal Backbone:** Pyramid TCN for capturing multi-scale temporal patterns from weather station data.
3.  **Physics-Informed Logic:** Explicit integration of Solar Zenith Angle (SZA) and Azimuth, calculated via `pvlib`, to ground the model in physical reality.
4.  **Ladder Fusion:** A hierarchical fusion mechanism that combines visual and temporal features at multiple scales.

## Architecture

```mermaid
graph TD
    %% --- STYLING ---
    classDef imageEnc fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:black;
    classDef timeEnc fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:black;
    classDef fusion fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:black;
    classDef phys fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black;
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:black;

    %% --- INPUTS ---
    ASI["ASI Image<br>(512x512 RGB)"]:::imageEnc
    TS["Time Series<br>(60 mins × 5 vars)"]:::timeEnc
    Meta["Station Metadata<br>Lat 37.09, Lon -2.36"]:::phys

    %% --- PHYSICS LAYER ---
    subgraph "Physics Pre-processing"
        Calc["Calculate Solar Position (pvlib)"]:::phys
        AugTS["Augmented Time Series<br>GHI, Temp, Press, SZA, Azimuth"]:::phys
    end

    %% --- VISUAL ENCODER ---
    subgraph "Visual Backbone (MambaVision)"
        direction TB
        Stem["Stem: Conv 3x3"]:::imageEnc
        
        S1_Blk["Stage 1 Blocks"]:::imageEnc
        S1_Out["Feat 1"]:::imageEnc
        S1_Down["Downsample"]:::imageEnc
        
        S2_Blk["Stage 2 Blocks"]:::imageEnc
        S2_Out["Feat 2"]:::imageEnc
        S2_Down["Downsample"]:::imageEnc

        S3_Blk["Stage 3 Hybrid"]:::imageEnc
        S3_Out["Feat 3"]:::imageEnc
        S3_Down["Downsample"]:::imageEnc

        S4_Blk["Stage 4 Hybrid"]:::imageEnc
        S4_Out["Feat 4"]:::imageEnc
    end

    %% --- TEMPORAL ENCODER ---
    subgraph "Temporal Backbone (Pyramid TCN)"
        direction TB
        T_Emb["Embedding (7 channels)"]:::timeEnc
        
        T1["Branch 1: High Freq"]:::timeEnc
        T2["Branch 2: Daily Cycle"]:::timeEnc
        T3["Branch 3: Weather Sys"]:::timeEnc
        T4["Branch 4: Season Trend"]:::timeEnc
    end

    %% --- LADDER FUSION ---
    subgraph "Ladder Fusion"
        direction TB
        F1["Fusion I<br>Texture + Noise"]:::fusion
        F2["Fusion II<br>Shape + Daily"]:::fusion
        F3["Fusion III<br>Global + Weather"]:::fusion
        F4["Fusion IV<br>Semantic + Trend"]:::fusion
    end

    %% --- OUTPUT HEAD ---
    Pool["Global Avg Pooling"]:::output
    Concat["Concatenation"]:::output
    MLP["Regressor Head"]:::output
    Pred["GHI Prediction"]:::output

    %% --- CONNECTIONS ---
    Meta --> Calc --> AugTS
    TS --> AugTS
    
    ASI --> Stem --> S1_Blk
    S1_Blk --> S1_Out --> F1
    S1_Blk --> S1_Down --> S2_Blk
    
    S2_Blk --> S2_Out --> F2
    S2_Blk --> S2_Down --> S3_Blk

    S3_Blk --> S3_Out --> F3
    S3_Blk --> S3_Down --> S4_Blk

    S4_Blk --> S4_Out --> F4

    AugTS --> T_Emb 
    T_Emb --> T1 --> F1
    T_Emb --> T2 --> F2
    T_Emb --> T3 --> F3
    T_Emb --> T4 --> F4

    F1 & F2 & F3 & F4 --> Pool --> Concat --> MLP --> Pred
```

## Data Pipeline

### 1. Metadata Parsing
The system reads `PSA_timeSeries_Metas.csv` to extract station coordinates:
- **Latitude:** 37.0916
- **Longitude:** -2.3636
- **Altitude:** 490.587

### 2. Physics-Informed Feature Engineering
Using `pvlib`, we calculate:
- **Solar Zenith Angle (SZA)**
- **Solar Azimuth Angle**

These are appended to the raw weather data (GHI, DNI, DHI, Temp, Pressure) to form a 7-channel time series.

### 3. Night Filtering
Data points with **SZA > 85°** are strictly filtered out to prevent the model from learning trivial zeros during night time.

### 4. Image Matching
Images are matched to the nearest timestamp in the weather data. The system ensures a complete 60-minute history exists for each sample.

## Training

### Configuration
Edit `SolarMamba/config.yaml` to adjust parameters.

```yaml
training:
  epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 1.0e-2
```

### Execution
Run the training script from the `SolarMamba` directory:

```bash
python train.py
```

### Loss Function
**MSE Loss** is used to minimize the error between predicted and actual GHI.

### Metrics
- **RMSE:** Root Mean Square Error ($W/m^2$)
- **nRMSE:** Normalized RMSE (%)
- **Skill Score:** Improvement over persistence baseline.

## Dependencies

- `torch`, `torchvision`
- `pandas`, `numpy`
- `pvlib` (Critical for solar geometry)
- `timm` (For MambaVision dependencies)
- `mamba_ssm` (For Mamba blocks)
- `PyYAML`
- `tqdm`
- `matplotlib`
- `einops`

## Quick Start

1. **Data Setup:**
   - **Local:** Run `python utils/mock_data.py` to generate dummy data.
   - **Server:** Ensure data is at `/storage2/CV_Irradiance`.
   - Update `config.yaml` to set `env: "local"` or `env: "server"`.

2. **Run Training:**

```bash
cd SolarMamba
python train.py --output_dir ./checkpoints
```

3. **Run Evaluation:**

```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pth
```
