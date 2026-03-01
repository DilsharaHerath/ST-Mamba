import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from datetime import datetime


# -----------------------------
# Helpers
# -----------------------------
def _decode_dates(arr):
    arr = np.asarray(arr)
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            s = x.decode("utf-8")
        else:
            s = str(x)
        out.append(s)
    return np.array(out, dtype=object)


def _load_holiday_set(holiday_txt_path):
    holidays = set()
    if holiday_txt_path and os.path.exists(holiday_txt_path):
        with open(holiday_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                holidays.add(line[:8])
    return holidays


def _time_features_from_dates(date_strings):
    """
    4 features:
      sin_time_slot, cos_time_slot, sin_day_of_week, cos_day_of_week
    plus holiday flag added later -> total core = 5
    """
    N = len(date_strings)
    feats = np.zeros((N, 4), dtype=np.float32)

    for i, ds in enumerate(date_strings):
        ymd = ds[:8]
        tt = ds[8:]  # slot index (1..48)

        try:
            slot = int(tt)
        except Exception:
            slot = 1

        slot = max(1, min(48, slot))
        ang_slot = 2 * math.pi * (slot - 1) / 48.0
        feats[i, 0] = math.sin(ang_slot)
        feats[i, 1] = math.cos(ang_slot)

        try:
            dt = datetime.strptime(ymd, "%Y%m%d")
            dow = dt.weekday()  # 0..6
        except Exception:
            dow = 0

        ang_dow = 2 * math.pi * dow / 7.0
        feats[i, 2] = math.sin(ang_dow)
        feats[i, 3] = math.cos(ang_dow)

    return feats


def _ensure_nchw(flow):
    flow = np.asarray(flow)
    if flow.ndim != 4:
        raise ValueError(f"Expected 4D flow tensor, got {flow.shape}")

    if flow.shape[1] in (1, 2, 3, 4):
        return flow
    if flow.shape[-1] in (1, 2, 3, 4):
        return np.transpose(flow, (0, 3, 1, 2))
    return flow


# -----------------------------
# Dataset
# -----------------------------
class TaxiBJH5Dataset(Dataset):
    """
    Returns:
      x_seq   : (T, 2, 32, 32)
      ext_seq : (T, E)     (E can be 0 if disabled)
      targets : (H, 2, 32, 32)   multi-horizon targets
      meta    : dict with dates
    """

    def __init__(
        self,
        h5_flow_paths,
        meteorology_h5_path=None,
        holiday_txt_path=None,
        sequence_length=12,
        horizons=(1, 2, 3, 4),   # horizons in "steps" (TaxiBJ is 30-min per step)
        normalize=True,
        use_external=True,
        stats=None,
    ):
        self.sequence_length = int(sequence_length)
        self.horizons = [int(h) for h in horizons]
        self.normalize = bool(normalize)
        self.use_external = bool(use_external)

        flows_all, dates_all = [], []

        for p in h5_flow_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Flow h5 not found: {p}")

            with h5py.File(p, "r") as f:
                flow_key = None
                date_key = None
                for k in f.keys():
                    lk = k.lower()
                    if lk == "data":
                        flow_key = k
                    if lk == "date":
                        date_key = k

                if flow_key is None:
                    raise KeyError(f"No 'data' dataset found in {p}. Keys: {list(f.keys())}")
                if date_key is None:
                    raise KeyError(f"No 'date' dataset found in {p}. Keys: {list(f.keys())}")

                flow = f[flow_key][:]
                date = f[date_key][:]

            flow = _ensure_nchw(flow).astype(np.float32)
            date = _decode_dates(date)

            flows_all.append(flow)
            dates_all.append(date)

        self.flow = np.concatenate(flows_all, axis=0)  # (N,2,32,32)
        self.date = np.concatenate(dates_all, axis=0)  # (N,)

        # ---- external features ----
        self.ext = None
        if self.use_external:
            ext_parts = []

            # 4 time features
            ext_parts.append(_time_features_from_dates(self.date))  # (N,4)

            # holiday flag -> 5th feature
            holidays = _load_holiday_set(holiday_txt_path)
            hol = np.array([1.0 if d[:8] in holidays else 0.0 for d in self.date], dtype=np.float32).reshape(-1, 1)
            ext_parts.append(hol)

            # meteorology (optional)
            if meteorology_h5_path and os.path.exists(meteorology_h5_path):
                with h5py.File(meteorology_h5_path, "r") as f:
                    met_key = None
                    met_date_key = None
                    for k in f.keys():
                        lk = k.lower()
                        if lk in ("meteorology", "met"):
                            met_key = k
                        if lk == "date":
                            met_date_key = k

                    if met_key is not None and met_date_key is not None:
                        met = np.asarray(f[met_key][:], dtype=np.float32)
                        met_date = _decode_dates(f[met_date_key][:])

                        met_flat = met.reshape(met.shape[0], -1)
                        if len(met_date) == len(self.date) and np.all(met_date == self.date):
                            met_aligned = met_flat
                        else:
                            idx_map = {d: i for i, d in enumerate(met_date)}
                            met_aligned = np.zeros((len(self.date), met_flat.shape[1]), dtype=np.float32)
                            for i, d in enumerate(self.date):
                                j = idx_map.get(d, None)
                                if j is not None:
                                    met_aligned[i] = met_flat[j]

                        ext_parts.append(met_aligned.astype(np.float32))

            self.ext = np.concatenate(ext_parts, axis=1).astype(np.float32)  # (N,E)

        # ---- indices ----
        self.N = self.flow.shape[0]
        max_h = max(self.horizons)
        min_i = self.sequence_length - 1
        max_i = self.N - 1 - max_h
        if max_i < min_i:
            raise ValueError(
                f"Not enough timesteps: N={self.N}, sequence_length={self.sequence_length}, max_horizon={max_h}"
            )
        self.indices = list(range(min_i, max_i + 1))

        # ---- normalization ----
        C = self.flow.shape[1]  # should be 2
        if self.normalize:
            if stats is not None:
                mean = np.asarray(stats["mean"], dtype=np.float32).reshape(C, 1, 1)
                std = np.asarray(stats["std"], dtype=np.float32).reshape(C, 1, 1)
            else:
                mean = self.flow.mean(axis=(0, 2, 3)).astype(np.float32).reshape(C, 1, 1)
                std = self.flow.std(axis=(0, 2, 3)).astype(np.float32).reshape(C, 1, 1)
                std = np.maximum(std, 1e-6)
            self.mean = mean
            self.std = std
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        start = i - (self.sequence_length - 1)
        end = i + 1  # exclusive

        x_seq = self.flow[start:end]  # (T,2,32,32)

        # multi-horizon targets: (H,2,32,32)
        ys = []
        for h in self.horizons:
            ys.append(self.flow[i + h])
        targets = np.stack(ys, axis=0)

        if self.normalize:
            x_seq = (x_seq - self.mean) / self.std
            targets = (targets - self.mean) / self.std

        if self.ext is None:
            ext_seq = np.zeros((self.sequence_length, 0), dtype=np.float32)
        else:
            ext_seq = self.ext[start:end]

        meta = {
            "hist_dates": self.date[start:end].tolist(),
            "target_dates": [self.date[i + h] for h in self.horizons],
            "horizons": self.horizons,
        }

        return (
            torch.from_numpy(x_seq).float(),       # (T,2,32,32)
            torch.from_numpy(ext_seq).float(),     # (T,E)
            torch.from_numpy(targets).float(),     # (H,2,32,32)
            meta,
        )


def get_data_loaders(config):
    # Resolve root dir
    if config["env"] == "server":
        root_dir = config["data"].get("server_root", "")
        if not root_dir:
            raise ValueError("env is 'server' but 'server_root' is not set in config.")
    elif config["env"] == "colab":
        root_dir = config["data"]["colab_root"]
    else:
        root_dir = config["data"]["local_root"]

    data_cfg = config["data"]

    seq_len = int(data_cfg.get("sequence_length", 12))
    normalize = bool(data_cfg.get("normalize", True))
    use_external = bool(data_cfg.get("use_external", True))

    # Horizons: for TaxiBJ these are "steps" (1 step = 30 min)
    # To match Solar's "4 horizons" behavior, default to 4 steps.
    horizons = config.get("model", {}).get("horizons", [1, 2, 3, 4])
    horizons = [int(h) for h in horizons]

    years = data_cfg.get("years", [2013, 2014, 2015, 2016])
    year_files = []
    for y in years:
        p = os.path.join(root_dir, f"BJ{str(y)[-2:]}_M32x32_T30_InOut.h5")
        if os.path.exists(p):
            year_files.append(p)
    if not year_files:
        raise FileNotFoundError(f"No BJ??_M32x32_T30_InOut.h5 files found in: {root_dir}")

    meteorology_h5 = os.path.join(root_dir, "BJ_Meteorology.h5")
    holiday_txt = os.path.join(root_dir, "BJ_Holiday.txt")

    full_tmp = TaxiBJH5Dataset(
        h5_flow_paths=year_files,
        meteorology_h5_path=meteorology_h5 if os.path.exists(meteorology_h5) else None,
        holiday_txt_path=holiday_txt if os.path.exists(holiday_txt) else None,
        sequence_length=seq_len,
        horizons=horizons,
        normalize=normalize,
        use_external=use_external,
        stats=None,
    )

    total_len = len(full_tmp)
    training_cfg = config.get("training", {})
    val_split = float(training_cfg.get("val_split", data_cfg.get("val_split", 0.1)))
    test_split = float(training_cfg.get("test_split", data_cfg.get("test_split", 0.1)))
    train_split = 1.0 - val_split - test_split
    if train_split <= 0:
        raise ValueError("Invalid split: train_split must be > 0. Check val_split/test_split.")

    train_len = int(train_split * total_len)
    val_len = int(val_split * total_len)
    test_len = total_len - train_len - val_len

    indices = list(range(total_len))
    train_idx = indices[:train_len]
    val_idx = indices[train_len:train_len + val_len]
    test_idx = indices[train_len + val_len:]

    # Use train stats for all splits
    stats = None
    if normalize:
        C = full_tmp.flow.shape[1]
        stats = {"mean": full_tmp.mean.reshape(C).copy(), "std": full_tmp.std.reshape(C).copy()}

    full = TaxiBJH5Dataset(
        h5_flow_paths=year_files,
        meteorology_h5_path=meteorology_h5 if os.path.exists(meteorology_h5) else None,
        holiday_txt_path=holiday_txt if os.path.exists(holiday_txt) else None,
        sequence_length=seq_len,
        horizons=horizons,
        normalize=normalize,
        use_external=use_external,
        stats=stats,
    )

    train_dataset = torch.utils.data.Subset(full, train_idx)
    val_dataset = torch.utils.data.Subset(full, val_idx)
    test_dataset = torch.utils.data.Subset(full, test_idx)

    num_workers = int(data_cfg.get("num_workers", 0))
    if config.get("env") == "colab":
        num_workers = min(num_workers, 2)

    batch_size = int(data_cfg.get("batch_size", 8))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader