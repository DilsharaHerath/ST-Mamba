# SolarMamba Configuration

env: "server" # Options: "local", "server"

data:
  local_root: "../mock_data_storage"
  server_root: "/storage2/CV_Irradiance"
  dataset_type: "distorted" # Options: "distorted" (Original) or "undistorted" (Corrected)
  months: ['09', '10', '11']
  
  # Sampling
  sampling_rate_sec: 30
  sequence_length: 60 # 60 steps * 30s = 30 minutes history
  
  # Image
  image_size: 512
  
  # Loader
  batch_size: 16
  num_workers: 4

model:
  visual_backbone: "mamba_vision_B"
  temporal_channels: 7 # k*, Temp, Pressure, SZA, Azimuth, sin_hour, cos_hour
  horizons: [1, 5, 10, 15] # Minutes
  pretrained_weights: "../weights/mambavision_b_1k.pth"

training:
  epochs: 50
  learning_rate: 1.0e-4 # Increased for breakout
  weight_decay: 1.0e-3 # Decreased for freedom
  val_split: 0.2 # Last 20% chronologically
  seed: 42


