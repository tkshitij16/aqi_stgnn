import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_yaml, ensure_dir, save_scalers, get_logger
from src.dataset import STTDataset, make_split_epochs
from src.model import STTGNN

log = get_logger("train")

def set_seed(s):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_master(cfg):
    p = cfg["data"]["master_csv_path"]
    df = pd.read_csv(p, low_memory=False)
    df[cfg["data"]["time_col"]] = pd.to_numeric(df[cfg["data"]["time_col"]], errors="coerce").astype("int64")
    df[cfg["data"]["node_col"]] = df[cfg["data"]["node_col"]].astype(str)
    log.info(f"[Data] Loaded: {p} | rows={len(df):,}")
    return df

def split_df(df, cfg):
    time_col = cfg["data"]["time_col"]
    (tr0,tr1),(va0,va1),(te0,te1) = make_split_epochs(cfg)
    train = df[(df[time_col]>=tr0)&(df[time_col]<=tr1)].copy()
    val   = df[(df[time_col]>=va0)&(df[time_col]<=va1)].copy()
    test  = df[(df[time_col]>=te0)&(df[time_col]<=te1)].copy()
    log.info(f"[Split] Train={len(train):,} | Val={len(val):,} | Test={len(test):,}")
    return train, val, test

def collate(batch): return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if cfg["training"]["device"]=="cuda" and torch.cuda.is_available() else "cpu")
    log.info(f"[Device] Using {device}")

    df = load_master(cfg)
    train_df, val_df, _ = split_df(df, cfg)

    train_ds = STTDataset(train_df, cfg, fit_scalers=True, verbose=True)
    val_ds   = STTDataset(
        val_df,
        cfg,
        dyn_scaler=train_ds.dyn_scaler,
        stat_scaler=train_ds.stat_scaler,
        target_stats=train_ds.target_stats,
        fit_scalers=False,
        verbose=False,
    )

    model = STTGNN(
        dyn_dim=train_ds.Ddyn,
        static_dim=train_ds.Dstat,
        gnn_hidden=cfg["model"]["gnn_hidden"],
        gru_hidden=cfg["model"]["gru_hidden"],
        gnn_layers=cfg["model"]["gnn_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    huber = torch.nn.HuberLoss(delta=1.0, reduction='none')
    task = cfg["training"]["task"]
    bs = cfg["training"]["batch_hours"]

    tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate)

    best_val = float("inf")
    ensure_dir(cfg["outputs"]["model_file"])
    ensure_dir(cfg["outputs"]["scaler_file"])

    for epoch in range(1, cfg["training"]["epochs"]+1):
        model.train()
        tr_losses = []
        for batch in tr_loader:
            opt.zero_grad()
            loss_all = 0.0
            for data in batch:
                data = data.to(device)
                aqi_hat, pm_hat = model(data.x_dyn, data.x_stat, data.edge_index, data.edge_weight)
                loss = 0.0
                if task in ("aqi","both") and data.mask_aqi.any():
                    loss += huber(aqi_hat[data.mask_aqi], data.y_aqi[data.mask_aqi]).mean()
                if task in ("pm25","both") and data.mask_pm.any():
                    loss += huber(pm_hat[data.mask_pm], data.y_pm[data.mask_pm]).mean()
                loss_all += loss
            loss_all.backward()
            opt.step()
            tr_losses.append(float(loss_all.item()))

        model.eval()
        with torch.no_grad():
            va_losses = []
            for batch in va_loader:
                loss_all = 0.0
                for data in batch:
                    data = data.to(device)
                    aqi_hat, pm_hat = model(data.x_dyn, data.x_stat, data.edge_index, data.edge_weight)
                    if task in ("aqi","both") and data.mask_aqi.any():
                        loss_all += huber(aqi_hat[data.mask_aqi], data.y_aqi[data.mask_aqi]).mean()
                    if task in ("pm25","both") and data.mask_pm.any():
                        loss_all += huber(pm_hat[data.mask_pm], data.y_pm[data.mask_pm]).mean()
                va_losses.append(float(loss_all.item()))
            va_avg = float(np.mean(va_losses)) if va_losses else np.nan

        log.info(f"[Epoch {epoch:03d}] train={np.mean(tr_losses):.4f} | val={va_avg:.4f}")

        if va_avg < best_val:
            best_val = va_avg
            torch.save(model.state_dict(), cfg["outputs"]["model_file"])
            save_scalers(
                cfg["outputs"]["scaler_file"],
                train_ds.dyn_scaler,
                train_ds.stat_scaler,
                train_ds.target_stats,
            )
            log.info(f"[Checkpoint] Saved best model → {cfg['outputs']['model_file']} | scalers → {cfg['outputs']['scaler_file']}")

if __name__ == "__main__":
    main()
