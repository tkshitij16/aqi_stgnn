import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import (
    load_yaml,
    ensure_dir,
    load_scalers,
    chicago_epoch_to_local_str,
    get_logger,
)
from src.dataset import STTDataset
from src.model import STTGNN

log = get_logger("predict")

def load_master(cfg):
    p = cfg["data"]["master_csv_path"]
    df = pd.read_csv(p, low_memory=False)
    df[cfg["data"]["time_col"]] = pd.to_numeric(df[cfg["data"]["time_col"]], errors="coerce").astype("int64")
    df[cfg["data"]["node_col"]] = df[cfg["data"]["node_col"]].astype(str)
    log.info(f"[Data] Loaded for predict: {p} | rows={len(df):,}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if cfg["training"]["device"]=="cuda" and torch.cuda.is_available() else "cpu")
    log.info(f"[Device] Using {device}")

    df = load_master(cfg)
    ds, ss = load_scalers(cfg["outputs"]["scaler_file"])
    full_ds = STTDataset(df, cfg, dyn_scaler=ds, stat_scaler=ss, fit_scalers=False, verbose=False)

    model = STTGNN(
        dyn_dim=full_ds.Ddyn,
        static_dim=full_ds.Dstat,
        gnn_hidden=cfg["model"]["gnn_hidden"],
        gru_hidden=cfg["model"]["gru_hidden"],
        gnn_layers=cfg["model"]["gnn_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(cfg["outputs"]["model_file"], map_location=device))
    model.eval()

    loader = DataLoader(
        full_ds,
        batch_size=cfg["training"]["batch_hours"],
        shuffle=False,
        collate_fn=lambda x: x,
    )
    rows = []
    with torch.no_grad():
        for batch in loader:
            for data in batch:
                data = data.to(device)
                aqi_hat, pm_hat = model(data.x_dyn, data.x_stat, data.edge_index, data.edge_weight)
                aqi_np = aqi_hat.cpu().numpy()
                pm_np  = pm_hat.cpu().numpy()
                ids = full_ds.nodes["node_id"].tolist()
                for i in range(len(ids)):
                    rows.append(
                        dict(
                            epoch=int(data.t_end),
                            node_id=ids[i],
                            aqi_hat=float(aqi_np[i]),
                            pm25_hat=float(pm_np[i]),
                        )
                    )

    out = pd.DataFrame(rows)

    # Attach static node metadata (lat/lon) for downstream mapping convenience
    nodes = full_ds.nodes.rename(columns={"node_id": "node_id", "lat": "lat", "lon": "lon"})
    out = out.merge(nodes, on="node_id", how="left")
    missing_ll = out["lat"].isna() | out["lon"].isna()
    if missing_ll.any():
        log.warning(
            "[Predict] %d prediction rows are missing lat/lon after join; consider updating master CSV.",
            int(missing_ll.sum()),
        )

    # Provide consistent naming for downstream dashboards (aqi_pred / pm25_pred)
    out["aqi_pred"] = out["aqi_hat"]
    out["pm25_pred"] = out["pm25_hat"]

    pred_csv = cfg["outputs"]["pred_csv"]
    ensure_dir(pred_csv)
    out.to_csv(pred_csv, index=False)

    out2 = out.copy()
    out2["time_local"] = chicago_epoch_to_local_str(out2["epoch"])
    pred_with_time = cfg["outputs"]["pred_csv_with_time"]
    ensure_dir(pred_with_time)
    out2.to_csv(pred_with_time, index=False)

    log.info(
        "[Predict] Wrote predictions → %s (rows=%d)",
        pred_csv,
        len(out),
    )

    # Write master copy augmented with predictions for dashboard workflows
    master_with_preds = cfg["outputs"].get("master_with_preds")
    if master_with_preds:
        preds_for_merge = out[["epoch", "node_id", "aqi_pred", "pm25_pred"]]
        enriched_master = df.merge(
            preds_for_merge,
            on=[cfg["data"]["time_col"], cfg["data"]["node_col"]],
            how="left",
        )
        ensure_dir(master_with_preds)
        enriched_master.to_csv(master_with_preds, index=False)
        log.info(
            "[Predict] Wrote master with predictions → %s", master_with_preds
        )

if __name__ == "__main__":
    main()
