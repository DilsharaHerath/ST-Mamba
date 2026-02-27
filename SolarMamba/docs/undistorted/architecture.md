# **SolarMamba Architecture**

SolarMamba is a hybrid physics–deep learning model built to forecast solar irradiance by fusing three information streams:

1. **Physics Engine (pvlib):** Determines the sun's geometry and clear-sky irradiance.
2. **Visual Encoder (MambaVision-B):** Learns spatial–semantic cloud structure from ASI images.
3. **Temporal Encoder (Pyramid TCN):** Learns multi-scale temporal dynamics from meteorological history.
4. **Ladder Fusion:** Injects temporal knowledge into visual features through a hierarchical gating mechanism.

The model predicts the **Clear Sky Index**:

$$
k^* = \frac{\mathrm{GHI}}{\mathrm{GHI}_{\mathrm{cs}}}
$$

The final forecast is reconstructed as:

$$
\widehat{\mathrm{GHI}} = k^* \cdot \mathrm{GHI}_{\mathrm{cs}}
$$

---

# **1. Full Architecture Diagram**

```mermaid
---
config:
  layout: elk
---
flowchart LR

%% ===========================
%% PHYSICS ENGINE
%% ===========================
subgraph subGraph0["Physics Engine (pvlib)"]
    Calc["Solar Position<br>SZA, Azimuth"]
    CS["Clear Sky GHI<br>Ineichen Model"]
    Norm["Normalize Target<br>k* = GHI / GHI_cs"]
end

%% ===========================
%% VISUAL ENCODER
%% ===========================
subgraph subGraph1["Visual Encoder (MambaVision-B)"]
    S1["Stage 1<br>128 → 256 Channels"]
    S2["Stage 2<br>256 → 512 Channels"]
    S3["Stage 3<br>512 → 1024 Channels"]
    S4["Stage 4<br>1024 → 1024 Channels"]
end

%% ===========================
%% TEMPORAL ENCODER
%% ===========================
subgraph subGraph2["Temporal Encoder (Pyramid TCN)"]
    T1["Branch 1<br>k = 3"]
    T2["Branch 2<br>k = 5"]
    T3["Branch 3<br>k = 7"]
    T4["Branch 4<br>k = 9"]
end

%% ===========================
%% FUSION LADDERS
%% ===========================
subgraph subGraph3["Ladder Fusion (Sigmoid Gating)"]
    F1(("Gate"))
    F2(("Gate"))
    F3(("Gate"))
    F4(("Gate"))
end

%% ===========================
%% DATA INPUTS
%% ===========================
Meta["PSA Metadata<br>Lat / Lon / Time"] --> Calc & CS
Hist["7-Channel History<br>GHI, Temp, Pressure,<br>SZA, Azimuth, sin(h), cos(h)"] --> T1 & T2 & T3 & T4

CS --> Norm
Img["ASI Image<br>(512×512)"] --> S1

%% ===========================
%% VISUAL PIPE
%% ===========================
S1 --> S2 & F1
S2 --> S3 & F2
S3 --> S4 & F3
S4 --> F4

%% ===========================
%% TEMPORAL → FUSION
%% ===========================
T1 --> F1
T2 --> F2
T3 --> F3
T4 --> F4

%% ===========================
%% AGGREGATION + HEAD
%% ===========================
F1 --> Concat["Aggregator<br>(Concat + GAP)"]
F2 --> Concat
F3 --> Concat
F4 --> Concat

Concat --> PredK["Predict k*<br>(MLP Head)"]
PredK --> Recon["Reconstruct GHI<br>GHI = k* × GHI_cs"]
Recon --> Final["Final Forecast"]

%% ===========================
%% STYLE
%% ===========================
Calc:::phys
CS:::phys
Norm:::phys

S1:::deep
S2:::deep
S3:::deep
S4:::deep

T1:::temp
T2:::temp
T3:::temp
T4:::temp

F1:::fusion
F2:::fusion
F3:::fusion
F4:::fusion

Meta:::phys
Hist:::temp
Img:::deep

Concat:::output
PredK:::output
Recon:::output
Final:::output

classDef phys fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black
classDef deep fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:black
classDef temp fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:black
classDef fusion fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:black
classDef output fill:#e0f2f1,stroke:#004d40,stroke-width:2px,color:black

```

---

## **2. Components**

## **2.1 Visual Encoder — MambaVision-B**

A hierarchical 4-stage MambaVision backbone.

- Stage outputs (visual resolutions shrink, channels expand):
  - Stage 1: $H/4 \times W/4$ → **256 channels**
  - Stage 2: $H/8 \times W/8$ → **512 channels**
  - Stage 3: $H/16 \times W/16$ → **1024 channels**
  - Stage 4: $H/32 \times W/32$ → **1024 channels**

Each stage produces a feature map $V_i$ used in fusion.

## **2.2 Temporal Encoder — Pyramid TCN**

Input sequence:

$$X \in \mathbb{R}^{B \times T \times 7}$$

Channels include:

$$[GHI, T, P, \text{SZA}, \text{Azimuth}, \sin(h), \cos(h)]$$

After a linear embedding to 128 dimensions, four Conv1d branches with different receptive fields capture multi-scale temporal patterns:

| Branch | Kernel Size | Temporal Scale | Output |
|--------|-------------|----------------|--------|
| 1 | 3 | short-range | 128-D |
| 2 | 5 | mid-short | 128-D |
| 3 | 7 | mid-long | 128-D |
| 4 | 9 | long-range | 128-D |

Each branch applies Conv1d → ReLU → Global Average Pooling, producing **pooled embeddings** (not feature maps).

Output:

$$[t_1, t_2, t_3, t_4],\quad t_i \in \mathbb{R}^{B \times 128}$$
---

## **3. Ladder Fusion — Temporal-Conditioned Visual Gating**

Given visual map:

$$V \in \mathbb{R}^{B \times C_v \times H \times W}$$

And temporal vector:

$$T \in \mathbb{R}^{B \times 128}$$

The gate is computed as:

$$g = \sigma(W_t T) \in \mathbb{R}^{B \times C_v}$$

Broadcast spatially to match visual dimensions:

$$g \in \mathbb{R}^{B \times C_v \times H \times W}$$

Fusion (residual gated amplification):

$$V_{\text{out}} = V \odot g + V$$

This injects temporal context without altering spatial resolution or channel dimensions.
---

# **4. Prediction Head**

1. Global Average Pool on the fused visual outputs
2. Concatenate:

$$256 + 512 + 1024 + 1024 = 2816$$

3. MLP:

$$\mathrm{LN} \rightarrow \mathrm{Linear}(2816 \rightarrow 512) \rightarrow \mathrm{GELU} \rightarrow \mathrm{Dropout}(0.5) \rightarrow \mathrm{Linear}(512 \rightarrow 4)$$

Outputs: forecasts at horizons $$[1, 5, 10, 15]$$ minutes.

---


# **5. Detailed Component Diagrams **

---

## **5.1 Pyramid TCN **

Each branch produces a **128-dimensional pooled embedding** through Global Average Pooling, not feature maps. These embeddings are used to modulate corresponding visual stages.

```mermaid
graph TD
    style Input fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:black
    style Embed fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style Permute fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style B1 fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style B2 fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style B3 fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style B4 fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style R1 fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:black
    style R2 fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:black
    style R3 fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:black
    style R4 fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:black
    style P1 fill:#e0f2f1,stroke:#004d40,stroke-width:3px,color:black
    style P2 fill:#e0f2f1,stroke:#004d40,stroke-width:3px,color:black
    style P3 fill:#e0f2f1,stroke:#004d40,stroke-width:3px,color:black
    style P4 fill:#e0f2f1,stroke:#004d40,stroke-width:3px,color:black
    style Output fill:#e0f2f1,stroke:#004d40,stroke-width:3px,color:black
    
    Input["Weather Sequence<br>(B,T,7)"] --> Embed["Linear Embedding to 128"]
    Embed --> Permute["Permute to (B,Emb,T)"]
    
    subgraph Parallel_Branches["Parallel Branches"]
        Permute --> B1["Conv1d k=3"] --> R1["ReLU"] --> P1["Global Avg Pool"]
        Permute --> B2["Conv1d k=5"] --> R2["ReLU"] --> P2["Global Avg Pool"]
        Permute --> B3["Conv1d k=7"] --> R3["ReLU"] --> P3["Global Avg Pool"]
        Permute --> B4["Conv1d k=9"] --> R4["ReLU"] --> P4["Global Avg Pool"]
    end
    
    P1 & P2 & P3 & P4 --> Output["List of 4 vectors<br>(B,128)"]
```

---

## **5.2 Ladder Fusion**

```mermaid
flowchart TB
 subgraph Inputs["Inputs"]
        V["Visual Feature<br>(B,Cv,H,W)"]
        T["Temporal Vector<br>(B,Ct)"]
  end
    T -- <br> --> Linear["Linear Projection to Cv"]
    Linear -- <br> --> Reshape["Reshape & Broadcast to B,Cv,1,1"]
    Reshape -- <br> --> Sigmoid["Sigmoid Activation"]
    Sigmoid -- <br> --> Gate(("Gate"))
    V -- <br> --> Mult["Element-wise Multiply"] & Add["Residual Add:<br>V + V*Gate"]
    Gate -- <br> --> Mult
    Mult -- <br> --> Add
    Add -- <br> --> Output["Fused Feature Map"]

    style Linear fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style Reshape fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style Sigmoid fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style Gate fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,color:black
    style Mult fill:#FFE0B2,stroke:#FF6D00,stroke-width:3px,color:black
    style Add fill:#e0f2f1,stroke:#004d40,stroke-width:3px,color:black
    style Output fill:#e0f2f1,stroke:#004d40,stroke-width:3px,color:black
    style Inputs fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:black
    linkStyle 0 stroke:#000000,fill:none
    linkStyle 1 stroke:#000000,fill:none
    linkStyle 2 stroke:#000000
    linkStyle 3 stroke:#000000,fill:none
    linkStyle 4 stroke:#000000,fill:none
    linkStyle 5 stroke:#000000,fill:none
    linkStyle 6 stroke:#000000,fill:none
    linkStyle 7 stroke:#000000,fill:none
    linkStyle 8 stroke:#000000,fill:none
```

---

## **5.3 Mamba Mixer Block (Exact Mermaid Code)**

```mermaid
---
config:
  layout: dagre
---
flowchart LR
 subgraph Main_Branch["Main Branch"]
        Conv1_x["Conv1d"]
        Split{"Split"}
        SiLU_x["SiLU"]
        SSM["Selective Scan<br>SSM"]
  end
 subgraph Gating_Branch["Gating Branch"]
        Conv1_z["Conv1d"]
        SiLU_z["SiLU"]
  end
    Input["Input Tensor"] --> Norm["LayerNorm"]
    Norm --> Linear_In["Linear Projection"]
    Linear_In --> Split
    Split -- x --> Conv1_x
    Conv1_x --> SiLU_x
    SiLU_x --> SSM
    Split -- z --> Conv1_z
    Conv1_z --> SiLU_z
    SSM --> Concat["Concatenation"]
    SiLU_z --> Concat
    Concat --> Linear_Out["Linear Projection"]
    Linear_Out --> Output["Output Tensor"]

    style Conv1_x fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style Split fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,color:black
    style SiLU_x fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:black
    style SSM fill:#e0f2f1,stroke:#004d40,stroke-width:3px,color:black
    style Conv1_z fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style SiLU_z fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:black
    style Input fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:black
    style Norm fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style Linear_In fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style Concat fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,color:black
    style Linear_Out fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black
    style Output fill:#e0f2f1,stroke:#004d40,stroke-width:3px,color:black
```


