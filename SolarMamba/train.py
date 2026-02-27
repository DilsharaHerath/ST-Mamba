import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse



# Add parent directory to path to allow importing mambavision
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ladder_net import MambaLadder
from data_loader import get_data_loaders



def train():

    parser = argparse.ArgumentParser(description='Train SolarMamba')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save checkpoints')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

       

    # --- DIRECTORY SETUP ---

    dataset_type = config['data'].get('dataset_type')
    base_output_dir = os.path.join("Results", "checkpoints", dataset_type)

   
    # checkpoint_dir structure: Results/checkpoints/{dataset_type}/
    checkpoint_dir = base_output_dir

    #good model
    good_models_dir = os.path.join("Results","Good models", dataset_type)

       

    # Create output directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(good_models_dir):
        os.makedirs(good_models_dir)

    print(f"Training on '{dataset_type}' dataset.")
    print(f"Saving checkpoints to: {checkpoint_dir}")
    print(f"Saving best models to: {good_models_dir}")

    # Check Data Existence (Basic check on root)
    if config['env'] == 'server':
        root = config['data'].get('server_root', '')
        if not root:
            raise ValueError(
                "env is 'server' but 'server_root' is not set in config. "
                "Create a local config_server.yaml that adds this key."
            )
    elif config['env'] == 'colab':
        root = config['data']['colab_root']
    else:
        root = config['data']['local_root']


    if not os.path.exists(root):
        # If mock data generation is needed, we might want to warn
        print(f"Warning: Root directory {root} does not exist. Ensure data is present or run mock_data.py.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # --- DATASET STATISTICS ---
    full_dataset = train_loader.dataset.dataset  # unwrap Subset → SolarDataset
    total_timestamps = len(full_dataset.df)
    train_samples = len(train_loader.dataset)
    val_samples   = len(val_loader.dataset)
    test_samples  = len(test_loader.dataset)
    print("\n" + "="*55)
    print("  DATASET STATISTICS")
    print("="*55)
    print(f"  Total weather timestamps (daytime, all years): {total_timestamps:>8,}")
    print(f"  Matched samples — Train  (2017–2020):          {train_samples:>8,}")
    print(f"  Matched samples — Val    (2021):               {val_samples:>8,}")
    print(f"  Matched samples — Test   (2022):               {test_samples:>8,}")
    print("="*55 + "\n")

    # Model
    # Use colab-specific weights path when running in Colab.
    if config.get('env') == 'colab':
        model_path = config['model'].get('colab_pretrained_weights',
                     config['model'].get('pretrained_weights', None))
    else:
        model_path = config['model'].get('pretrained_weights', None)
    model = MambaLadder(pretrained=True, model_path=model_path).to(device)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    
    # Separate parameters into two groups
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "visual_backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    # Define base LR from config
    base_lr = float(config['training']['learning_rate'])
    
    optimizer = optim.AdamW([
        {
            'params': backbone_params, 
            'lr': base_lr * 0.01  # Try 0.1x (5e-6) or even 0.01x
        },
        {
            'params': head_params, 
            'lr': base_lr        # Keep original (5e-5)
        }
    ], weight_decay=float(config['training']['weight_decay']))
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

   

    # Training Loop
    epochs = config['training']['epochs']
    best_val_rmse = float('inf')

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for images, weather_seq, targets, _ in pbar:
            images = images.to(device)
            weather_seq = weather_seq.to(device)
            targets = targets.to(device) # Shape: (B, 4)

            optimizer.zero_grad()
            outputs = model(images, weather_seq) # Shape: (B, 4)
            loss = criterion(outputs, targets)

        
            # --- SAFETY CLAMPS ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at Epoch {epoch+1}. Skipping batch.")
                continue

            loss.backward()

            # Gradient Clipping (Prevents explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # ---------------------

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

           
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds_k = []
        all_targets_k = []
        all_ghi_cs = []

       
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, weather_seq, targets, ghi_cs in pbar_val:
                images = images.to(device)
                weather_seq = weather_seq.to(device)
                targets = targets.to(device)

                outputs = model(images, weather_seq)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            
                all_preds_k.append(outputs.cpu().numpy())
                all_targets_k.append(targets.cpu().numpy())
                all_ghi_cs.append(ghi_cs.numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Metrics on Reconstructed GHI (Average across all horizons for summary)
        all_preds_k = np.concatenate(all_preds_k, axis=0) # (N, 4)
        all_targets_k = np.concatenate(all_targets_k, axis=0) # (N, 4)
        all_ghi_cs = np.concatenate(all_ghi_cs, axis=0) # (N, 4)

       

        # Reconstruct GHI
        # Handle Subset wrapper to access dataset attributes

        if isinstance(train_loader.dataset, torch.utils.data.Subset):
            dataset = train_loader.dataset.dataset
        else:
            dataset = train_loader.dataset

        std_k = dataset.std['k_index']
        mean_k = dataset.mean['k_index']

        pred_k_raw = (all_preds_k * std_k) + mean_k
        target_k_raw = (all_targets_k * std_k) + mean_k

        # Then calculate GHI
        pred_ghi = pred_k_raw * all_ghi_cs
        actual_ghi = target_k_raw * all_ghi_cs

        rmse = np.sqrt(np.mean((pred_ghi - actual_ghi)**2))
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val RMSE (Avg): {rmse:.4f}")

       
        # Step Scheduler
        scheduler.step(rmse)

        # 1. Always save current epoch to checkpoints folder
        ckpt_name = f"model_ep{epoch+1}_rmse{rmse:.4f}.pth"
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, ckpt_name))

        if rmse < best_val_rmse:  # Changed from avg_val_loss
            best_val_rmse = rmse  # Track RMSE instead
            # Save specific best file
            best_name = f"best_model_ep{epoch+1}_rmse{rmse:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(good_models_dir, best_name))

            print(f"Saved New Best Model (RMSE: {best_val_rmse:.4f}) to {good_models_dir}")



    # Visualize one batch after training
    # print("Generating visualization...")
    # visualize_batch(model, val_loader) # Visualization needs update for multi-horizon, skipping for now or update later

def visualize_batch(model, loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    images, weather_seq, targets, ghi_cs = next(iter(loader))

    images = images.to(device)

    weather_seq = weather_seq.to(device)

    with torch.no_grad():

        preds_k = model(images, weather_seq)

    # Visualize first sample in batch

    img = images[0].cpu().permute(1, 2, 0).numpy()

    # Un-normalize image for display

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    img = std * img + mean

    img = np.clip(img, 0, 1)


    # k* trend

    k_trend = weather_seq[0, :, 0].cpu().numpy() # k_index is col 0

    pred_k_val = preds_k[0].item()

    actual_k_val = targets[0].item()

   

    ghi_cs_val = ghi_cs[0].item()

    pred_ghi = pred_k_val * ghi_cs_val

    actual_ghi = actual_k_val * ghi_cs_val

   

    plt.figure(figsize=(12, 5))

   

    plt.subplot(1, 2, 1)

    plt.imshow(img)

    plt.title("ASI Image (Masked)")

    plt.axis('off')

   

    plt.subplot(1, 2, 2)

    plt.plot(k_trend, label='Past 60m k*')

    plt.scatter(60, pred_k_val, color='red', label='Predicted k*', marker='x', s=100)

    plt.scatter(60, actual_k_val, color='green', label='Actual k*', marker='o')

    plt.title(f"Forecast\nPred GHI: {pred_ghi:.1f}, Actual: {actual_ghi:.1f} W/m2")

    plt.legend()

    plt.grid(True)

    plt.ylim(0, 1.3)

   

    plt.tight_layout()

    plt.savefig("visualization.png")

    print("Saved visualization.png")



if __name__ == "__main__":

    train()
