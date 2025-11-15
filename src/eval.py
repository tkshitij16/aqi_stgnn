import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import (
    load_yaml, ensure_dir, load_scalers, rmse, mae, r2, get_logger
)
from src.dataset import STTDataset, make_split_epochs
from src.model import STTGNN

log = get_logger("eval")

# ---------------- utility loaders ----------------
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
    tr = df[(df[time_col]>=tr0)&(df[time_col]<=tr1)].copy()
    va = df[(df[time_col]>=va0)&(df[time_col]<=va1)].copy()
    te = df[(df[time_col]>=te0)&(df[time_col]<=te1)].copy()
    log.info(f"[Split] Train={len(tr):,} | Val={len(va):,} | Test={len(te):,}")
    return tr, va, te

# ---------------- evaluation core ----------------
def _invert(values, stats):
    if not stats:
        return values
    mu = stats.get("mean", 0.0)
    sd = stats.get("std", 1.0)
    if not np.isfinite(sd) or sd == 0.0:
        sd = 1.0
    return values * sd + mu


def evaluate_dataset(ds, cfg, device, model=None):
    """Return metrics dict for a (possibly empty) STTDataset."""
    if ds is None or len(ds) == 0:
        return dict(
            aqi_rmse=np.nan,
            aqi_mae=np.nan,
            aqi_r2=np.nan,
            pm25_rmse=np.nan,
            pm25_mae=np.nan,
            pm25_r2=np.nan,
            n_hours=0,
        )

    # Build model on the fly if not provided (uses ds dims)
    if model is None:
        model = STTGNN(
            dyn_dim=ds.Ddyn,
            static_dim=ds.Dstat,
            gnn_hidden=cfg["model"]["gnn_hidden"],
            gru_hidden=cfg["model"]["gru_hidden"],
            gnn_layers=cfg["model"]["gnn_layers"],
            dropout=cfg["model"]["dropout"]
        ).to(device)
        model.load_state_dict(torch.load(cfg["outputs"]["model_file"], map_location=device))
    model.eval()

    loader = DataLoader(ds, batch_size=cfg["training"]["batch_hours"], shuffle=False, collate_fn=lambda x: x)

    aqi_pred, aqi_true = [], []
    pm_pred, pm_true = [], []
    with torch.no_grad():
        for batch in loader:
            for data in batch:
                data = data.to(device)
                aqi_hat, pm_hat = model(data.x_dyn, data.x_stat, data.edge_index, data.edge_weight)

                if data.mask_aqi.any():
                    mask = data.mask_aqi.cpu().numpy()
                    aqi_pred.append(aqi_hat.cpu().numpy()[mask])
                    aqi_true.append(data.y_aqi.cpu().numpy()[mask])

                if data.mask_pm.any():
                    mask_pm = data.mask_pm.cpu().numpy()
                    pm_pred.append(pm_hat.cpu().numpy()[mask_pm])
                    pm_true.append(data.y_pm.cpu().numpy()[mask_pm])

    stats = ds.target_stats if hasattr(ds, "target_stats") else {}

    def summarise(pred_list, true_list, key):
        if not pred_list:
            return dict(rmse=np.nan, mae=np.nan, r2=np.nan)
        yh = np.concatenate(pred_list)
        yt = np.concatenate(true_list)
        yh = _invert(yh, stats.get(key))
        yt = _invert(yt, stats.get(key))
        return dict(
            rmse=rmse(yh, yt),
            mae=mae(yh, yt),
            r2=r2(yh, yt),
        )

    aqi_metrics = summarise(aqi_pred, aqi_true, "aqi")
    pm_metrics = summarise(pm_pred, pm_true, "pm25")

    return dict(
        aqi_rmse=aqi_metrics["rmse"],
        aqi_mae=aqi_metrics["mae"],
        aqi_r2=aqi_metrics["r2"],
        pm25_rmse=pm_metrics["rmse"],
        pm25_mae=pm_metrics["mae"],
        pm25_r2=pm_metrics["r2"],
        n_hours=len(ds),
    )

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if (cfg["training"]["device"]=="cuda" and torch.cuda.is_available()) else "cpu")
    log.info(f"[Device] Using {device}")

    # master + splits
    df = load_master(cfg)
    _, val_df, test_df = split_df(df, cfg)

    # load scalers (fit on train during training)
    ds_s, ss_s, target_stats = load_scalers(cfg["outputs"]["scaler_file"])

    # build datasets; guard against empty splits
    val_ds  = STTDataset(val_df,  cfg, dyn_scaler=ds_s, stat_scaler=ss_s, target_stats=target_stats, fit_scalers=False, verbose=False) if len(val_df)  else None
    test_ds = STTDataset(test_df, cfg, dyn_scaler=ds_s, stat_scaler=ss_s, target_stats=target_stats, fit_scalers=False, verbose=False) if len(test_df) else None

    # Build a single model once (use val_ds dims if available, else skip test)
    model = None
    if val_ds is not None and len(val_ds) > 0:
        model = STTGNN(
            dyn_dim=val_ds.Ddyn,
            static_dim=val_ds.Dstat,
            gnn_hidden=cfg["model"]["gnn_hidden"],
            gru_hidden=cfg["model"]["gru_hidden"],
            gnn_layers=cfg["model"]["gnn_layers"],
            dropout=cfg["model"]["dropout"]
        ).to(device)
        model.load_state_dict(torch.load(cfg["outputs"]["model_file"], map_location=device))

    # evaluate
    log.info("[Eval] Running validation metrics …")
    val_metrics = evaluate_dataset(val_ds, cfg, device, model=model)
    log.info(
        "[Val] AQI RMSE={val_metrics['aqi_rmse']:.3f} MAE={val_metrics['aqi_mae']:.3f} R2={val_metrics['aqi_r2']:.3f} | "
        "PM25 RMSE={val_metrics['pm25_rmse']:.3f} MAE={val_metrics['pm25_mae']:.3f} R2={val_metrics['pm25_r2']:.3f} "
        f"(hours={val_metrics['n_hours']})"
    )

    if test_ds is None or len(test_ds) == 0:
        log.warning("[Eval] Test split is empty — writing NaN metrics for test.")
        test_metrics = dict(rmse=np.nan, mae=np.nan, r2=np.nan, n_hours=0)
    else:
        # if we didn't have a model yet (e.g., val also empty), build using test dims
        if model is None:
            model = STTGNN(
                dyn_dim=test_ds.Ddyn,
                static_dim=test_ds.Dstat,
                gnn_hidden=cfg["model"]["gnn_hidden"],
                gru_hidden=cfg["model"]["gru_hidden"],
                gnn_layers=cfg["model"]["gnn_layers"],
                dropout=cfg["model"]["dropout"]
            ).to(device)
            model.load_state_dict(torch.load(cfg["outputs"]["model_file"], map_location=device))
        log.info("[Eval] Running test metrics …")
        test_metrics = evaluate_dataset(test_ds, cfg, device, model=model)
        log.info(
            "[Test] AQI RMSE={test_metrics['aqi_rmse']:.3f} MAE={test_metrics['aqi_mae']:.3f} R2={test_metrics['aqi_r2']:.3f} | "
            "PM25 RMSE={test_metrics['pm25_rmse']:.3f} MAE={test_metrics['pm25_mae']:.3f} R2={test_metrics['pm25_r2']:.3f} "
            f"(hours={test_metrics['n_hours']})"
        )

    # write table
    rows = [
        dict(
            split="val",
            aqi_rmse=val_metrics["aqi_rmse"],
            aqi_mae=val_metrics["aqi_mae"],
            aqi_r2=val_metrics["aqi_r2"],
            pm25_rmse=val_metrics["pm25_rmse"],
            pm25_mae=val_metrics["pm25_mae"],
            pm25_r2=val_metrics["pm25_r2"],
            n_hours=val_metrics["n_hours"],
        ),
        dict(
            split="test",
            aqi_rmse=test_metrics["aqi_rmse"],
            aqi_mae=test_metrics["aqi_mae"],
            aqi_r2=test_metrics["aqi_r2"],
            pm25_rmse=test_metrics["pm25_rmse"],
            pm25_mae=test_metrics["pm25_mae"],
            pm25_r2=test_metrics["pm25_r2"],
            n_hours=test_metrics["n_hours"],
        ),
    ]
    out = pd.DataFrame(rows)
    out_path = cfg["outputs"]["metrics_csv"]
    ensure_dir(out_path); out.to_csv(out_path, index=False)
    log.info(f"[Metrics] Wrote → {out_path}\n{out}")

if __name__ == "__main__":
    main()
