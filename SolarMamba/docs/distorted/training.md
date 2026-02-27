# Training Instructions

## Prerequisites
- Python 3.8+
- PyTorch
- pvlib
- pandas
- matplotlib

## Configuration
Edit `SolarMamba/config.yaml` to adjust hyperparameters:
```yaml
training:
  epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 0.01
```

## Running Training

# Clear cache
rm -rf /storage2/CV_Irradiance/scripts/Mamba-Irradiance-V2/SolarMamba/models/__pycache__

# Run Training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml --output_dir ./checkpoints/1min | tee training_log.txt

### Standard Training (1-minute horizon)
```bash
cd SolarMamba
python train.py --horizon 1 --output_dir ./checkpoints/1min
```

### Multi-Horizon Training
To train models for different forecast horizons (e.g., 15 minutes), specify the `--horizon` argument:
```bash
python train.py --horizon 15 --output_dir ./checkpoints/15min
```

## Evaluation

After training, use the evaluation script to benchmark performance and run ablation studies.

```bash
python evaluate.py --checkpoints ./checkpoints/1min/best_model.pth ./checkpoints/15min/best_model.pth --horizons 1 15
```
```bash
python evaluate.py --config config.yaml --checkpoint "Results/Good models/distorted/best_model.pth"
```

This will:
1.  Calculate RMSE and Skill Score for each horizon.
2.  Run **Modality Ablation** (Image-Blind vs. Time-Blind) to quantify the contribution of visual vs. temporal features.
3.  Generate a performance plot `evaluation_results.png`.

## Loss Function
- **Type:** MSE Loss
- **Target:** Clear Sky Index ($k^*$)
- **Reasoning:** $k^*$ is a normalized measure of cloudiness (0-1.2), making it a more stable target than raw GHI (0-1000+).

## Optimization
- **Optimizer:** AdamW
- **Learning Rate:** 1e-4
- **Weight Decay:** 0.01
- **Scheduler:** CosineAnnealingWarmRestarts (T_0=10, T_mult=2)

## Metrics
- **Loss:** MSE on $k^*$.
- **RMSE:** Root Mean Square Error on reconstructed GHI ($W/m^2$).

