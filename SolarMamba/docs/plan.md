# Experiment Plan


## Objective

- The objective of this experiment is to evaluate the performance and efficiency of the SolarMamba V2 system under various test setups. 

- We have a All Sky Image dataset and corresponding Global Horizontal Irradiance (GHI) measurements. The goal is to forecast GHI values at multiple future time horizons (1, 5, 10, and 15 minutes) using the SolarMamba architecture.

- Main components to be tested:
  - Visual Backbone: MambaVision-B
  - Temporal Backbone: PyramidTCN
  - Fusion Mechanism: LadderFusion
  - Prediction Head: Multi-scale pooled MLP

- Main challenges:
  
  - Selecting extra rich features to feed into the model. Now we provide,
    1. Visual Data:
        - ASI Images (now we control the sampling rate at 30seconds and we vary the sequence length)
        Problems to consider: 
            - What is the optimal sequence length?
            - What is the optimal sampling rate?
            - What is the optimal image resolution?
            - what data augmentation techniques to use during training?(e.g., random cropping, flipping, color jittering)
            - What pre-processing techniques to apply to the images?(e.g., resizing, normalization)
            - How to handle varying lighting conditions and weather patterns in the images?(e.g., normalization, histogram equalization)
            - How to extract meaningful features from the images that correlate with GHI values?(e.g., using pre-trained models, custom feature extractors, FFTs, wavelets, wsts, nsst, xlets, curvelets, shearlets, etc.)
            - How to effectively combine spatial and temporal information from the images?(e.g., using CNNs, RNNs, attention mechanisms)
            - cloud detection and segmentation techniques to isolate relevant regions in the images?
            - What architectures to use for the visual backbone?(e.g., ResNet, EfficientNet, MambaVision, etc.)
            - What transfer learning strategies to employ when using pre-trained models?(e.g., fine-tuning, feature extraction)
            - Novel techniques like self-supervised learning or contrastive learning to improve feature extraction?
            (e.g., SimCLR, MoCo, BYOL)
            - physics informed neural networks to incorporate domain knowledge about solar irradiance and cloud dynamics?
            (e.g., using physical models as constraints or regularizers during training, hybrid models combining data-driven and physics-based approaches, fourier neural operators, neural operators, etc.)
            - How to evaluate the quality of the extracted features?(e.g., using visualization techniques, correlation analysis)


    2. Time series data:
        - Historical GHI values
        - DNI values
        - DHI values
        - Azimuth and Zenith angles of the sun
        - Time of Day ( as sin(h) and cos(h))
        - temperature
        - atmospheric pressure
        Problems to consider: 
            - What is the optimal sequence length for temporal data?
            - What additional meteorological variables can improve forecasting accuracy?
            - How to handle missing or noisy data in the time series?
            - What normalization or scaling techniques to apply to the temporal data?
            - How to effectively combine multiple temporal features?(e.g., feature engineering, dimensionality reduction)
            - What architectures to use for the temporal backbone?(e.g., TCN, LSTM, GRU, Transformer, etc.)
            - How to capture long-term dependencies in the temporal data?(e.g., attention mechanisms, dilated convolutions)
            - What fusion strategies to employ for combining visual and temporal features?(e.g., early fusion, late fusion, cross-attention)
            - Novel techniques like self-supervised learning or contrastive learning to improve temporal feature extraction?
            (e.g., SimCLR, MoCo, BYOL)
            - physics informed neural networks to incorporate domain knowledge about solar irradiance and cloud dynamics?
            (e.g., using physical models as constraints or regularizers during training, hybrid models combining data-driven and physics-based approaches, fourier neural operators, neural operators, etc.)
            - Time series encoders to better capture temporal patterns?(e.g., temporal convolutional networks, recurrent neural networks, transformers, timesformers, etc.)
            - How to evaluate the quality of the extracted temporal features?(e.g., using visualization techniques, correlation analysis)

General problems to consider:
    - What is the optimal batch size?
    - What is the optimal learning rate and optimizer?
    - What is the optimal number of epochs?
    - What is the optimal loss function?
    - What is the optimal evaluation metrics?
    - How to effectively combine visual and temporal data?
    - How to handle missing or corrupted data in the sequences?
    - How to optimize model architecture for both accuracy and efficiency?

  - Ensuring effective fusion of visual and temporal features.
  - Achieving robust performance across all forecast horizons.
  - Maintaining computational efficiency for real-time applications.
  - Skill improvement over baseline persistence models.



## Experiment Setup

- Datasets:
  - Use the All Sky Image dataset along with corresponding GHI measurements.
  - Split the data into training, validation, and test sets.(val_split : 0.2)
- Model Configurations:
    - Baseline Model: MambaVision-B only (visual backbone).
    - Temporal Model: PyramidTCN only (temporal backbone).
    - Full Model: SolarMamba with LadderFusion.
- Training:
    - Train each model configuration for a fixed number of epochs (e.g., 50 epochs).
    - Use early stopping based on validation loss to prevent overfitting.
    - Experiment with different learning rates and batch sizes.
- Evaluation:
    - Evaluate models on the test set using RMSE, MAE, MBE, R², and Skill Score.
    - Analyze performance across all forecast horizons (1, 5, 10, 15 minutes).
    - Compare results against baseline persistence models.

    

