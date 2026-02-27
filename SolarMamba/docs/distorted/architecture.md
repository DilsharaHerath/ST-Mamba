# SolarMamba Architecture

## Overview
SolarMamba is a hybrid physical-deep learning model designed for solar irradiance forecasting. It combines a visual encoder (MambaVision) with a temporal encoder (Pyramid TCN) using a gated fusion mechanism.

## Diagram

```mermaid
graph TD
    %% --- STYLING ---
    classDef phys fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black;
    classDef deep fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:black;
    classDef fusion fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:black;
    classDef output fill:#e0f2f1,stroke:#004d40,stroke-width:2px,color:black;

    %% --- INPUTS ---
    Img["ASI Image<br>(512×512)"]:::deep
    Meta["PSA Metadata<br>Lat / Lon / Time"]:::phys
    Hist["Met History<br>GHI, Temp, Pressure"]:::phys

    %% --- PHYSICS ENGINE ---
    subgraph "Physics Engine (pvlib)"
        Calc["Solar Position<br>SZA, Azimuth"]:::phys
        CS["Clear Sky GHI<br>Ineichen Model"]:::phys
        Norm["Normalize Target<br>k* = GHI / GHI_cs"]:::phys
    end

    %% --- ENCODERS ---
    subgraph "Visual Encoder (MambaVision)"
        S1["Stage 1<br>Texture"]:::deep
        S2["Stage 2<br>Shape"]:::deep
        S3["Stage 3<br>Global"]:::deep
        S4["Stage 4<br>Semantic"]:::deep
    end

    subgraph "Temporal Encoder (Pyramid TCN)"
        T1["Branch 1<br>Noise"]:::phys
        T2["Branch 2<br>Daily"]:::phys
        T3["Branch 3<br>System"]:::phys
        T4["Branch 4<br>Trend"]:::phys
    end

    %% --- GATED FUSION ---
    subgraph "Ladder Fusion (Sigmoid Gating)"
        F1((Gate)):::fusion
        F2((Gate)):::fusion
        F3((Gate)):::fusion
        F4((Gate)):::fusion
    end

    %% --- FLOW ---
    Meta --> Calc
    Meta --> CS
    Hist --> Norm
    CS --> Norm
    
    Img --> S1 --> S2 --> S3 --> S4
    
    Calc --> T1 & T2 & T3 & T4
    Norm --> T1 & T2 & T3 & T4

    %% Cross Connections
    S1 --> F1
    T1 --> F1

    S2 --> F2
    T2 --> F2

    S3 --> F3
    T3 --> F3

    S4 --> F4
    T4 --> F4

    %% RECONSTRUCTION
    F1 & F2 & F3 & F4 --> Concat["Aggregator"]:::output
    Concat --> PredK["Predict k*"]:::output
    PredK --> Recon
    CS --> Recon
    Recon["Reconstruct GHI<br>GHI = k* × GHI_cs"]:::output --> Final["Final Forecast"]:::output
```

## Components

### 1. Visual Encoder
- **Backbone:** MambaVision-B
- **Features:** Extracted from Stages 1, 2, 3, and 4.

### 2. Temporal Encoder
- **Architecture:** Pyramid TCN
- **Branches:** 4 parallel branches with kernel sizes 3, 5, 7, 9.
- **Input:** 7 channels (k*, Temp, Pressure, SZA, Azimuth, sin_hour, cos_hour).

### 3. Ladder Fusion
- **Mechanism:** Residual Gating.
- **Logic:** The temporal vector is projected and passed through a sigmoid to create a gate. This gate modulates the visual features.
- **Formula:** $V_{out} = V_{in} \cdot \sigma(T_{proj}) + V_{in}$

### 4. Head
- **Structure:** Global Average Pooling -> Concatenation -> MLP -> Sigmoid -> Scale (1.2).
- **Output:** Clear Sky Index ($k^*$).

## Detailed Component Architectures

### 1. MambaVision Mixer Block
The core building block of the Visual Encoder. Unlike standard Transformers, it uses a State Space Model (SSM) for linear-time sequence modeling.

```mermaid
graph TD
    Input[Input Tensor] --> Norm[LayerNorm]
    Norm --> Linear_In[Linear Projection]
    Linear_In --> Split{Split}
    
    subgraph Main_Branch
        Split -- x --> Conv1_x[Conv1d]
        Conv1_x --> SiLU_x[SiLU]
        SiLU_x --> SSM["Selective Scan<br>SSM"]
    end
    
    subgraph Gating_Branch
        Split -- z --> Conv1_z[Conv1d]
        Conv1_z --> SiLU_z[SiLU]
    end
    
    SSM --> Concat[Concatenation]
    SiLU_z --> Concat
    Concat --> Linear_Out[Linear Projection]
    Linear_Out --> Output[Output Tensor]

```

### 2. Ladder Fusion Block
The mechanism for injecting physics-informed temporal context into the visual features. It uses a residual gating architecture.

```mermaid
graph TD
    subgraph Inputs
        V[Visual Feature Map]
        T[Temporal Vector]
    end
    
    T --> Linear[Linear Projection]
    Linear --> Reshape[Reshape & Broadcast]
    Reshape --> Sigmoid[Sigmoid Activation]
    Sigmoid --> Gate((Gate G))
    
    V --> Mult[Element-wise Multiply]
    Gate --> Mult
    
    Mult --> Add[Element-wise Add]
    V --> Add
    
    Add --> Output[Fused Feature]
```

### 3. Pyramid TCN (Temporal Encoder)
A multi-scale temporal encoder designed to capture weather patterns at different frequencies (noise, daily cycles, weather systems, trends).

```mermaid
graph TD
    Input[Weather Sequence] --> Embed[Linear Embedding]
    Embed --> Permute["Permute<br>(B,C,T)"]
    
    subgraph Parallel_Branches
        Permute --> B1[Conv1d k=3]
        Permute --> B2[Conv1d k=5]
        Permute --> B3[Conv1d k=7]
        Permute --> B4[Conv1d k=9]
        
        B1 --> R1[ReLU] --> P1[Global Avg Pool]
        B2 --> R2[ReLU] --> P2[Global Avg Pool]
        B3 --> R3[ReLU] --> P3[Global Avg Pool]
        B4 --> R4[ReLU] --> P4[Global Avg Pool]
    end
    
    P1 & P2 & P3 & P4 --> Output[List of 4 Vectors]

```
