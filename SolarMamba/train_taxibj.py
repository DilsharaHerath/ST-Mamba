import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ladder_net_taxibj import MambaLadder
from SolarMamba.data_loader_taxibj import get_data_loaders


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def evaluate(model, loader, device, denorm_stats=None):
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    all_pred, all_y = [], []

    for x_seq, ext_seq, targets, _ in loader:
        x_seq = x_seq.to(device)            # (B,T,2,32,32)
        ext_seq = ext_seq.to(device)        # (B,T,E)
        targets = targets.to(device)        # (B,H,2,32,32)

        pred = model(x_seq, ext_seq)        # (B,H,2,32,32)
        loss = criterion(pred, targets)
        total_loss += loss.item()

        all_pred.append(pred.detach().cpu().numpy())
        all_y.append(targets.detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))

    all_pred = np.concatenate(all_pred, axis=0)  # (N,H,2,32,32)
    all_y = np.concatenate(all_y, axis=0)

    rmse_norm = float(np.sqrt(np.mean((all_pred - all_y) ** 2)))

    rmse_raw = None
    if denorm_stats is not None:
        mean = np.asarray(denorm_stats["mean"], dtype=np.float32)
        std = np.asarray(denorm_stats["std"], dtype=np.float32)
        if mean.ndim == 1:
            mean = mean.reshape(1, 1, -1, 1, 1)   # (1,1,C,1,1)
        else:
            mean = mean.reshape(1, 1, mean.shape[0], 1, 1)
        if std.ndim == 1:
            std = std.reshape(1, 1, -1, 1, 1)
        else:
            std = std.reshape(1, 1, std.shape[0], 1, 1)

        pred_raw = all_pred * std + mean
        y_raw = all_y * std + mean
        rmse_raw = float(np.sqrt(np.mean((pred_raw - y_raw) ** 2)))

    return avg_loss, rmse_norm, rmse_raw


@torch.no_grad()
def save_debug_visualization(model, loader, device, save_dir, horizons, prefix="taxibj"):
    """
    Saves per-horizon figures:
      - last input (in/out)
      - target (in/out)
      - prediction (in/out)
    """
    ensure_dir(save_dir)
    model.eval()

    x_seq, ext_seq, targets, meta = next(iter(loader))
    x_seq = x_seq.to(device)
    ext_seq = ext_seq.to(device)
    targets = targets.to(device)

    pred = model(x_seq, ext_seq)

    # first sample in batch
    x0 = x_seq[0].detach().cpu()       # (T,2,32,32)
    y0 = targets[0].detach().cpu()     # (H,2,32,32)
    p0 = pred[0].detach().cpu()        # (H,2,32,32)

    last_in = x0[-1]                   # (2,32,32)

    for hi, h in enumerate(horizons):
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))

        # input
        im = axes[0, 0].imshow(last_in[0].numpy(), cmap="magma")
        axes[0, 0].set_title("Last Input - Inflow"); axes[0, 0].axis("off")
        plt.colorbar(im, ax=axes[0, 0])

        im = axes[0, 1].imshow(last_in[1].numpy(), cmap="magma")
        axes[0, 1].set_title("Last Input - Outflow"); axes[0, 1].axis("off")
        plt.colorbar(im, ax=axes[0, 1])

        # target
        im = axes[1, 0].imshow(y0[hi, 0].numpy(), cmap="magma")
        axes[1, 0].set_title(f"Target (h={h}) - Inflow"); axes[1, 0].axis("off")
        plt.colorbar(im, ax=axes[1, 0])

        im = axes[1, 1].imshow(y0[hi, 1].numpy(), cmap="magma")
        axes[1, 1].set_title(f"Target (h={h}) - Outflow"); axes[1, 1].axis("off")
        plt.colorbar(im, ax=axes[1, 1])

        # prediction
        im = axes[2, 0].imshow(p0[hi, 0].numpy(), cmap="magma")
        axes[2, 0].set_title(f"Pred (h={h}) - Inflow"); axes[2, 0].axis("off")
        plt.colorbar(im, ax=axes[2, 0])

        im = axes[2, 1].imshow(p0[hi, 1].numpy(), cmap="magma")
        axes[2, 1].set_title(f"Pred (h={h}) - Outflow"); axes[2, 1].axis("off")
        plt.colorbar(im, ax=axes[2, 1])

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{prefix}_h{h}.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

    print(f"✅ Saved debug plots to: {os.path.abspath(save_dir)}")


def train():
    parser = argparse.ArgumentParser(description="Train TaxiBJ with SolarMamba-style pipeline")
    parser.add_argument("--config", type=str, default="config_taxibj.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_type = config["data"].get("dataset_type", "taxibj")
    checkpoint_dir = os.path.join("Results", "checkpoints", dataset_type)
    good_models_dir = os.path.join("Results", "Good models", dataset_type)
    ensure_dir(checkpoint_dir)
    ensure_dir(good_models_dir)

    print(f"Training on '{dataset_type}' dataset.")
    print(f"Saving checkpoints to: {checkpoint_dir}")
    print(f"Saving best models to: {good_models_dir}")

    # Root check (same style)
    if config["env"] == "server":
        root = config["data"].get("server_root", "")
        if not root:
            raise ValueError("env is 'server' but 'server_root' is not set in config.")
    elif config["env"] == "colab":
        root = config["data"]["colab_root"]
    else:
        root = config["data"]["local_root"]

    if not os.path.exists(root):
        print(f"Warning: Root directory {root} does not exist. Check dataset path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # unwrap base dataset to get stats and ext dim
    base_ds = train_loader.dataset.dataset

    # horizons from config (steps)
    horizons = config.get("model", {}).get("horizons", [1, 2, 3, 4])
    horizons = [int(h) for h in horizons]
    H = len(horizons)

    # read ext dim
    x0, ext0, y0, _ = base_ds[0]
    E = ext0.shape[-1]
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS (TaxiBJ)")
    print("=" * 60)
    print(f"  Train samples: {len(train_loader.dataset):>8,}")
    print(f"  Val samples:   {len(val_loader.dataset):>8,}")
    print(f"  Test samples:  {len(test_loader.dataset):>8,}")
    print(f"  x_seq:         {tuple(x0.shape)} (T,2,32,32)")
    print(f"  ext_seq:       {tuple(ext0.shape)} (T,E)  E={E}")
    print(f"  targets:       {tuple(y0.shape)} (H,2,32,32)  H={H}")
    print("=" * 60 + "\n")

    denorm_stats = None
    if hasattr(base_ds, "mean") and base_ds.mean is not None:
        denorm_stats = {"mean": base_ds.mean.reshape(2), "std": base_ds.std.reshape(2)}

    # model: temporal_in_channels = flow_embed_dim + E
    flow_embed_dim = int(config["model"].get("flow_embed_dim", 32))
    temporal_in = flow_embed_dim + E

    model = MambaLadder(
        pretrained=False,
        model_path=None,
        task="taxibj",
        temporal_in_channels=temporal_in,
        taxibj_num_horizons=H,
        flow_embed_dim=flow_embed_dim,
        taxibj_out_h=32,
        taxibj_out_w=32,
    ).to(device)

    criterion = nn.MSELoss()

    base_lr = float(config["training"]["learning_rate"])
    wd = float(config["training"].get("weight_decay", 1e-4))

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    epochs = int(config["training"]["epochs"])
    best_val_rmse = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x_seq, ext_seq, targets, _ in pbar:
            x_seq = x_seq.to(device)
            ext_seq = ext_seq.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred = model(x_seq, ext_seq)
            loss = criterion(pred, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at Epoch {epoch+1}. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": float(loss.item())})

        avg_train_loss = train_loss / max(1, len(train_loader))

        val_loss, val_rmse_norm, val_rmse_raw = evaluate(model, val_loader, device, denorm_stats=denorm_stats)

        # choose score for scheduler/best: prefer raw if available
        score = val_rmse_raw if val_rmse_raw is not None else val_rmse_norm
        scheduler.step(score)

        print(
            f"Epoch {epoch+1}: TrainLoss {avg_train_loss:.6f} | "
            f"ValLoss {val_loss:.6f} | RMSE(norm) {val_rmse_norm:.6f}" +
            (f" | RMSE(raw) {val_rmse_raw:.6f}" if val_rmse_raw is not None else "")
        )

        # save epoch checkpoint
        ckpt_name = f"model_ep{epoch+1}_rmse{score:.6f}.pth"
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, ckpt_name))

        # best model
        if score < best_val_rmse:
            best_val_rmse = score
            best_name = f"best_model_ep{epoch+1}_rmse{score:.6f}.pth"
            torch.save(model.state_dict(), os.path.join(good_models_dir, best_name))
            print(f"✅ Saved New Best Model (RMSE: {best_val_rmse:.6f})")

        # periodic debug viz
        viz_every = int(config["training"].get("viz_every", 5))
        if (epoch + 1) == 1 or ((epoch + 1) % viz_every == 0):
            save_debug_visualization(
                model=model,
                loader=val_loader,
                device=device,
                save_dir="./debug_outputs",
                horizons=horizons,
                prefix=f"taxibj_ep{epoch+1}"
            )

    # final test
    test_loss, test_rmse_norm, test_rmse_raw = evaluate(model, test_loader, device, denorm_stats=denorm_stats)
    print("\nFINAL TEST:",
          f"Loss {test_loss:.6f} | RMSE(norm) {test_rmse_norm:.6f}" +
          (f" | RMSE(raw) {test_rmse_raw:.6f}" if test_rmse_raw is not None else ""))


if __name__ == "__main__":
    train()