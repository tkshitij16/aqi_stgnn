import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------- logging ----------
def get_logger(name: str = "aqi_stgnn", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

# ---------- io ----------
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_json(path, obj):
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ---------- time helpers ----------
def chicago_epoch_to_local_str(epoch_series, fmt="%Y-%m-%d %H:%M", tz="America/Chicago"):
    ts = pd.to_datetime(epoch_series.astype("int64"), unit="s", utc=True)
    return ts.dt.tz_convert(tz).dt.tz_localize(None).dt.strftime(fmt)

def to_epoch_local(chicago_str):
    tz="America/Chicago"
    ts=pd.Timestamp(chicago_str)
    loc=ts.tz_localize(tz, nonexistent="shift_forward", ambiguous=False)
    return int(loc.tz_convert("UTC").value//10**9)

# ---------- scalers ----------
def fit_scalers(train_dyn, train_stat):
    dyn_scaler = StandardScaler(with_mean=True, with_std=True)
    stat_scaler = StandardScaler(with_mean=True, with_std=True)
    dyn_scaler.fit(train_dyn)
    stat_scaler.fit(train_stat)
    return dyn_scaler, stat_scaler

def save_scalers(path, dyn_scaler, stat_scaler, target_stats=None):
    """Persist feature/target scaling parameters for later reuse.

    Parameters
    ----------
    path : str
        Destination NPZ file path.
    dyn_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for dynamic features.
    stat_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for static features.
    target_stats : Optional[Dict[str, Dict[str, float]]]
        Mapping like ``{"aqi": {"mean": μ, "std": σ}}`` describing how
        targets were standardised. Stored so that downstream prediction
        scripts can invert the transform back to physical units.
    """

    ensure_dir(path)
    payload = dict(
        dyn_mean=dyn_scaler.mean_,
        dyn_scale=dyn_scaler.scale_,
        stat_mean=stat_scaler.mean_,
        stat_scale=stat_scaler.scale_,
    )

    if target_stats:
        keys = list(target_stats.keys())
        payload["target_keys"] = np.array(keys, dtype="U16")
        payload["target_mean"] = np.array([target_stats[k]["mean"] for k in keys], dtype="float64")
        payload["target_scale"] = np.array([target_stats[k]["std"] for k in keys], dtype="float64")

    np.savez(path, **payload)


def load_scalers(path):
    """Load feature/target scaling parameters saved via :func:`save_scalers`."""

    z = np.load(path)
    ds = StandardScaler()
    ss = StandardScaler()
    ds.mean_ = z["dyn_mean"]
    ds.scale_ = z["dyn_scale"]
    ss.mean_ = z["stat_mean"]
    ss.scale_ = z["stat_scale"]

    target_stats = {}
    if "target_keys" in z.files:
        keys = z["target_keys"].tolist()
        means = z["target_mean"].astype(float)
        scales = z["target_scale"].astype(float)
        for key, mu, sd in zip(keys, means, scales):
            if not np.isfinite(sd) or sd == 0.0:
                sd = 1.0
            target_stats[str(key)] = {"mean": float(mu), "std": float(sd)}

    return ds, ss, target_stats

# ---------- metrics ----------
def rmse(yhat, y, mask=None):
    if mask is not None:
        yhat = yhat[mask]; y = y[mask]
    return float(np.sqrt(np.mean((yhat - y)**2))) if len(y) else np.nan

def mae(yhat, y, mask=None):
    if mask is not None:
        yhat = yhat[mask]; y = y[mask]
    return float(np.mean(np.abs(yhat - y))) if len(y) else np.nan

def r2(yhat, y, mask=None):
    if mask is not None:
        yhat = yhat[mask]; y = y[mask]
    if len(y)==0: return np.nan
    ss_res = np.sum((yhat - y)**2); ss_tot = np.sum((y - np.mean(y))**2)
    return float(1.0 - ss_res/(ss_tot + 1e-12))
