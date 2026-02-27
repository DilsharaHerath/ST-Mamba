import argparse
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from thop import profile, clever_format

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ladder_net import MambaLadder
from data_loader import get_data_loaders

# =============================================================================
# 1. RESEARCH-GRADE STATISTICAL FUNCTIONS
# =============================================================================

def diebold_mariano_test(real, pred1, pred2, h=1):
    """
    Performs the Diebold-Mariano test to check statistical significance 
    between two models (e.g., SolarMamba vs Persistence).
    """
    e1 = real - pred1
    e2 = real - pred2
    d = e1**2 - e2**2  # Loss differential (MSE)
    
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    if d_var == 0: return 0.0, 1.0 # Edge case
    
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat))) # Two-tailed
    
    return dm_stat, p_value

def calculate_ramp_metrics(targets, preds, threshold=50.0):
    """
    Evaluates ability to detect 'Ramp Events' (Sudden large changes).
    """
    true_ramps = np.abs(np.diff(targets)) > threshold
    pred_ramps = np.abs(np.diff(preds)) > threshold
    
    # Pad to match length
    true_ramps = np.append(true_ramps, False)
    pred_ramps = np.append(pred_ramps, False)
    
    TP = np.sum(true_ramps & pred_ramps)
    FP = np.sum(~true_ramps & pred_ramps)
    FN = np.sum(true_ramps & ~pred_ramps)
    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    csi = TP / (TP + FP + FN + 1e-6) # Critical Success Index
    
    return f1, csi, np.sum(true_ramps)

def calculate_grid_penalty(targets, preds):
    """
    Simulates a generic Grid Penalty score.
    Over-forecasting is penalized 2x more than Under-forecasting.
    """
    error = preds - targets
    penalty = np.where(error < 0, np.abs(error), 2.0 * np.abs(error))
    return np.mean(penalty)

# =============================================================================
# 2. MAIN EVALUATION LOGIC
# =============================================================================

def evaluate():
    parser = argparse.ArgumentParser(description='Evaluate SolarMamba (Research Grade)')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--benchmark', action='store_true', help='Run efficiency benchmark')
    parser.add_argument('--save_plots', action='store_true', default=True)
    args = parser.parse_args()

    # Setup Paths
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    dataset_type = config['data'].get('dataset_type', 'unknown')
    
    base_output_dir = os.path.join("Results")
    metrics_dir = os.path.join(base_output_dir, "evaluation metrics", dataset_type)
    plots_dir = os.path.join(base_output_dir, "plots", dataset_type)
    
    for d in [metrics_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Data for {dataset_type}...")
    _, _, test_loader = get_data_loaders(config)
    
    print(f"Loading Model: {args.checkpoint}")
    model = MambaLadder(pretrained=False).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        return
    model.eval()
    
    # # --- A. SPECIALIZED VISUALIZATIONS ---
    # history_path = os.path.join(metrics_dir, "training_history.json")
    # if os.path.exists(history_path): 
    #     plot_training_curve(history_path, plots_dir)

    # --- B. FULL INFERENCE WITH PHYSICS METADATA ---
    print("\n--- Running Full Model Inference ---")
    data_full = get_predictions_with_physics(model, test_loader, device, mode='full')
    metrics_full = calculate_advanced_metrics(data_full)
    
    # --- C. ABLATIONS ---
    print("\n--- Running Ablation: Image Blind ---")
    data_img = get_predictions_with_physics(model, test_loader, device, mode='image_blind')
    metrics_img = calculate_advanced_metrics(data_img)
    
    print("\n--- Running Ablation: Time Blind ---")
    data_time = get_predictions_with_physics(model, test_loader, device, mode='time_blind')
    metrics_time = calculate_advanced_metrics(data_time)

    # --- D. SAVING RESULTS ---
    all_results = {'Full_Model': metrics_full, 'Image_Blind': metrics_img, 'Time_Blind': metrics_time}
    json_path = os.path.join(metrics_dir, 'evaluation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved full metrics to {json_path}")
    
    print_summary_table(metrics_full)
    
    # --- E. GENERATE PLOTS ---
    print(f"\nGenerating Plots in {plots_dir}...")
    
    # 1. Physics Profiling (Error vs SZA/k*)
    plot_physics_error_profiling(data_full, plots_dir)
    
    # 2. Standard Diagnostics
    plot_timeseries_snapshot(data_full, plots_dir)
    plot_scatter_analysis(data_full, plots_dir)
    plot_error_distribution(data_full, plots_dir)
    plot_ablation_comparison(all_results, plots_dir)
    print("Plots saved.")

    run_efficiency_benchmark(model, device, metrics_dir)

# =============================================================================
# 3. DATA COLLECTION
# =============================================================================

def get_predictions_with_physics(model, dataloader, device, mode='full'):
    """
    Runs inference and captures: Preds, Targets, Persist, SZA, k_index.
    """
    horizons = [1, 5, 10, 15]
    # Initialize dictionary to store lists for each horizon
    res = {h: {'pred': [], 'target': [], 'persist': [], 'sza': [], 'k_index': []} for h in horizons}
    
    # Access dataset stats for Inverse Scaling
    if isinstance(dataloader.dataset, torch.utils.data.Subset): ds = dataloader.dataset.dataset
    else: ds = dataloader.dataset
    
    k_mean, k_std = float(ds.mean['k_index']), float(ds.std['k_index'])
    # Fallback if SZA stats missing (unlikely given dataloader)
    sza_mean = float(ds.mean.get('SZA', 0.0))
    sza_std = float(ds.std.get('SZA', 1.0))

    with torch.no_grad():
        for images, weather_seq, targets, ghi_cs in tqdm(dataloader, desc=f"Eval {mode}", leave=False):
            images, weather_gpu = images.to(device), weather_seq.to(device)
            
            # Physics Context (Before Ablation)
            sza_norm = weather_gpu[:, -1, 3].cpu().numpy()
            k_in_norm = weather_gpu[:, -1, 0].cpu().numpy()
            
            # Ablations
            if mode == 'image_blind': images = torch.zeros_like(images)
            if mode == 'time_blind': weather_gpu[:, :, :3] = 0 
            
            preds_norm = model(images, weather_gpu)
            
            # Inverse Scale Physics
            sza_real = (sza_norm * sza_std) + sza_mean
            k_in_real = (k_in_norm * k_std) + k_mean

            for i, h in enumerate(horizons):
                # Inverse Scale GHI
                p_real = (preds_norm[:, i].cpu().numpy() * k_std) + k_mean
                t_real = (targets[:, i].numpy() * k_std) + k_mean
                
                # Persistence
                per_real = k_in_real * ghi_cs[:, i].numpy()
                
                # Convert k* -> GHI
                p_ghi = p_real * ghi_cs[:, i].numpy()
                t_ghi = t_real * ghi_cs[:, i].numpy()
                per_ghi = per_real
                
                res[h]['pred'].extend(p_ghi)
                res[h]['target'].extend(t_ghi)
                res[h]['persist'].extend(per_ghi)
                res[h]['sza'].extend(sza_real)
                res[h]['k_index'].extend(k_in_real)

    # Convert lists to numpy arrays
    for h in horizons:
        for key in res[h]: res[h][key] = np.array(res[h][key])
        
    return res

def calculate_advanced_metrics(data):
    metrics = {}
    for h, d in data.items():
        p, t, per = d['pred'], d['target'], d['persist']
        
        # Mask NaNs
        mask = ~np.isnan(t) & ~np.isnan(p)
        p, t, per = p[mask], t[mask], per[mask]
        
        # --- Standard Metrics ---
        rmse = np.sqrt(mean_squared_error(t, p))
        mae = np.mean(np.abs(p - t))
        mbe = np.mean(p - t)
        
        # MABE (Mean Absolute Bias Error)
        mabe = np.mean(np.abs(p - t))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((t - p) / (t + 10.0))) * 100 
        
        r2 = r2_score(t, p)
        
        # Skill Score
        rmse_per = np.sqrt(mean_squared_error(t, per))
        skill = (1 - (rmse/(rmse_per + 1e-6))) * 100
        
        # --- Advanced Metrics ---
        dm_stat, p_value = diebold_mariano_test(t, per, p)
        f1_ramp, csi_ramp, _ = calculate_ramp_metrics(t, p, threshold=50.0)
        grid_cost = calculate_grid_penalty(t, p)
        
        metrics[h] = {
            'RMSE': float(rmse), 'MAE': float(mae), 'MBE': float(mbe), 
            'MABE': float(mabe), 'MAPE': float(mape),
            'R2': float(r2), 'Skill': float(skill),
            'DM_p_value': float(p_value), 'Ramp_CSI': float(csi_ramp), 'Grid_Cost': float(grid_cost)
        }
    return metrics

def print_summary_table(metrics):
    print("\n" + "="*115)
    print(f"{'Hor':<4} | {'RMSE':<7} | {'MABE':<7} | {'MAPE':<6} | {'Skill':<6} | {'DM (p)':<8} | {'Ramp CSI':<8} | {'GridCost':<8}")
    print("-" * 115)
    for h, m in metrics.items():
        sig = "*" if m['DM_p_value'] < 0.05 else "ns"
        print(f"{h:<4} | {m['RMSE']:<7.2f} | {m['MABE']:<7.2f} | {m['MAPE']:<6.1f} | {m['Skill']:<6.1f} | {m['DM_p_value']:<6.4f}{sig} | {m['Ramp_CSI']:<8.3f} | {m['Grid_Cost']:<8.2f}")
    print("="*115 + "\n")

# =============================================================================
# 4. PLOTTING FUNCTIONS
# =============================================================================

def plot_physics_error_profiling(data, output_dir):
    h = 15 # Analyze hardest horizon
    
    # Robust error check
    if len(data[h]['pred']) == 0: return

    df = pd.DataFrame({
        'Error': np.abs(data[h]['pred'] - data[h]['target']),
        'SZA': data[h]['sza'],
        'k_index': data[h]['k_index']
    })
    
    df['SZA_Bin'] = pd.cut(df['SZA'], bins=np.linspace(0, 90, 10))
    df['k_Bin'] = pd.cut(df['k_index'], bins=np.linspace(0, 1.2, 10))
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='SZA_Bin', y='Error', data=df, palette='Reds', errorbar=None)
    plt.xticks(rotation=45)
    plt.title(f'MABE vs Solar Zenith Angle (h={h})')
    plt.ylabel('MABE (W/m²)')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='k_Bin', y='Error', data=df, palette='Blues', errorbar=None)
    plt.xticks(rotation=45)
    plt.title(f'MABE vs Cloudiness (k*) (h={h})')
    plt.ylabel('MABE (W/m²)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'physics_error_profile.png'))
    plt.close()

def plot_timeseries_snapshot(data, output_dir, horizon_mins=[1, 15]):
    """
    Fixed slicing error: Accesses data[h]['pred'] instead of data[h]
    """
    limit = 400
    # Try to find a daytime slice (non-zero target)
    targets_15 = data[15]['target']
    daytime_idxs = np.where(targets_15 > 100)[0]
    
    if len(daytime_idxs) > limit:
        start = daytime_idxs[len(daytime_idxs)//2] # Mid-day
    else:
        start = 0
        
    plt.figure(figsize=(15, 8))
    
    for i, h in enumerate(horizon_mins):
        plt.subplot(len(horizon_mins), 1, i+1)
        
        # CORRECTED SLICING: Access 'pred' array inside the dictionary
        p = data[h]['pred'][start : start+limit]
        t = data[h]['target'][start : start+limit]
        
        plt.plot(t, 'k', label='Ground Truth', alpha=0.7)
        plt.plot(p, 'r' if i==0 else 'b', label='Prediction')
        plt.title(f'Forecast Horizon: {h} Minutes (MABE: {np.mean(np.abs(p-t)):.2f})')
        plt.ylabel('GHI (W/m²)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
    plt.xlabel('Time Steps')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeseries_snapshot.png'))
    plt.close()

def plot_training_curve(json_path, output_dir):
    with open(json_path, 'r') as f: history = json.load(f)
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_rmse'], color='orange', label='Val RMSE')
    plt.title('RMSE')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "training_curve.png"))
    plt.close()

def plot_scatter_analysis(data, output_dir):
    horizons = [1, 5, 10, 15]
    plt.figure(figsize=(12, 10))
    for i, h in enumerate(horizons):
        plt.subplot(2, 2, i+1)
        p, t = data[h]['pred'], data[h]['target']
        if len(p) > 5000:
            idx = np.random.choice(len(p), 5000, replace=False)
            p, t = p[idx], t[idx]
        plt.scatter(t, p, alpha=0.1, s=10, color='blue')
        mx = max(t.max(), p.max())
        plt.plot([0, mx], [0, mx], 'k--')
        plt.title(f'{h} min (R2={r2_score(t,p):.3f})')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_analysis.png'))
    plt.close()

def plot_error_distribution(data, output_dir):
    horizons = [1, 15]
    plt.figure(figsize=(12, 5))
    for i, h in enumerate(horizons):
        plt.subplot(1, 2, i+1)
        res = data[h]['pred'] - data[h]['target']
        plt.hist(res, bins=50, color='purple', alpha=0.7, density=True)
        plt.axvline(0, color='k', linestyle='--')
        plt.title(f'Error Dist ({h} min)')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()

def plot_ablation_comparison(results, output_dir):
    plt.figure(figsize=(12, 5))
    horizons = [1, 5, 10, 15]
    models = ['Full_Model', 'Image_Blind', 'Time_Blind']
    markers = ['o', 's', '^']
    
    plt.subplot(1, 2, 1)
    for m, mark in zip(models, markers):
        y = [results[m][h]['MABE'] for h in horizons]
        plt.plot(horizons, y, marker=mark, label=m)
    plt.title('MABE (MAE) Comparison')
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for m, mark in zip(models, markers):
        y = [results[m][h]['Skill'] for h in horizons]
        plt.plot(horizons, y, marker=mark, label=m)
    plt.title('Skill Comparison')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'ablation_chart.png'))
    plt.close()

def run_efficiency_benchmark(model, device, output_dir):
    print("\n--- Running Efficiency Benchmark ---")
    dummy_img = torch.randn(1, 3, 512, 512).to(device)
    dummy_seq = torch.randn(1, 40, 7).to(device)
    try:
        macs, params = profile(model, inputs=(dummy_img, dummy_seq), verbose=False)
        f_str, p_str = clever_format([macs, params], "%.3f")
        print(f"Params: {p_str}, FLOPs: {f_str}")
    except:
        macs, params = 0, 0
    
    iters = 100
    for _ in range(10): _ = model(dummy_img, dummy_seq)
    
    if device.type == 'cuda': torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters): _ = model(dummy_img, dummy_seq)
    if device.type == 'cuda': torch.cuda.synchronize()
    
    avg_t = (time.time()-start)/iters
    results = {"params": params, "flops": macs, "latency_ms": avg_t*1000, "fps": 1.0/avg_t}
    with open(os.path.join(output_dir, "efficiency_stats.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"FPS: {1.0/avg_t:.2f}")

if __name__ == "__main__":
    evaluate()
