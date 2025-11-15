import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GData

from src.utils import get_logger, to_epoch_local
from src.graph import PhysicsGraph

log = get_logger(__name__)

def make_split_epochs(cfg):
    tr = (to_epoch_local(cfg["data"]["train_hours"][0]),
          to_epoch_local(cfg["data"]["train_hours"][1]))
    va = (to_epoch_local(cfg["data"]["val_hours"][0]),
          to_epoch_local(cfg["data"]["val_hours"][1]))
    te = (to_epoch_local(cfg["data"]["test_hours"][0]),
          to_epoch_local(cfg["data"]["test_hours"][1]))
    return tr, va, te

class STTDataset(Dataset):
    """
    Snapshot per hour (end of W-hour window):
      - x_dyn:  [N, W, Ddyn]  (scaled by fitted scaler)
      - x_stat: [N, Dstat]
      - y_aqi / y_pm & masks
      - physics-aware graph at that hour
    """
    def __init__(
        self,
        df,
        cfg,
        dyn_scaler=None,
        stat_scaler=None,
        target_stats=None,
        fit_scalers=False,
        verbose=True,
    ):
        self.cfg = cfg
        dc = cfg["data"]
        self.time_col = dc["time_col"]
        self.node_col = dc["node_col"]
        self.dyn_cols = dc["dynamic_cols"]
        self.stat_cols = dc["static_cols"]
        self.pm_col   = dc["pm25_col"]
        self.aqi_col  = dc["aqi_col"]
        self.W        = dc["temporal_window"]

        self.df = df.copy()
        self.df[self.node_col] = self.df[self.node_col].astype(str)
        # force numeric on features/targets; blanks -> NaN
        for c in self.dyn_cols + self.stat_cols + [self.pm_col, self.aqi_col, dc["lat_col"], dc["lon_col"]]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        self.df = self.df.sort_values([self.time_col, self.node_col])

        # nodes table
        nodes = self.df[[dc["node_col"], dc["lat_col"], dc["lon_col"]]].drop_duplicates()
        nodes = nodes.rename(columns={dc["node_col"]:"node_id", dc["lat_col"]:"lat", dc["lon_col"]:"lon"})
        self.nodes = nodes.reset_index(drop=True)
        self.node2idx = {nid:i for i, nid in enumerate(self.nodes["node_id"].tolist())}
        self.N = len(self.nodes)

        # times
        self.times = np.sort(self.df[self.time_col].unique())
        self.valid_t_idx = np.arange(self.W-1, len(self.times))

        # tensors
        T = len(self.times); Ddyn = len(self.dyn_cols); Dstat = len(self.stat_cols)
        self.Ddyn, self.Dstat = Ddyn, Dstat

        dyn = np.full((T, self.N, Ddyn), np.nan, dtype=np.float32)
        aqi = np.full((T, self.N), np.nan, dtype=np.float32)
        pm  = np.full((T, self.N), np.nan, dtype=np.float32)

        # static: node means
        stat = np.zeros((self.N, Dstat), dtype=np.float32)
        g = self.df.groupby(self.node_col)
        for j, col in enumerate(self.stat_cols):
            if col in self.df.columns:
                m = g[col].mean()
                vals = self.nodes["node_id"].map(m)
                if vals.isna().all():
                    fallback = float(np.nanmean(pd.to_numeric(self.df[col], errors="coerce")))
                else:
                    fallback = float(np.nanmean(vals))
                if not np.isfinite(fallback):
                    fallback = 0.0
                vals = vals.fillna(fallback)
                stat[:, j] = vals.astype(np.float32).to_numpy()
            else:
                stat[:, j] = 0.0

        # dynamic + targets
        time2idx = {t:i for i,t in enumerate(self.times)}
        missing_dyn = [c for c in self.dyn_cols if c not in self.df.columns]
        if verbose and missing_dyn:
            log.warning(f"[Dataset] Missing dynamic columns: {missing_dyn}")

        for _, row in self.df.iterrows():
            ti = time2idx[row[self.time_col]]
            ni = self.node2idx[str(row[self.node_col])]
            for j, col in enumerate(self.dyn_cols):
                if col in self.df.columns and pd.notna(row[col]):
                    dyn[ti,ni,j] = np.float32(row[col])
            if self.aqi_col in self.df.columns and pd.notna(row[self.aqi_col]):
                aqi[ti,ni] = np.float32(row[self.aqi_col])
            if self.pm_col in self.df.columns and pd.notna(row[self.pm_col]):
                pm[ti,ni] = np.float32(row[self.pm_col])

        # coverage log
        if verbose:
            log.info("[Dataset] Dynamic coverage (non-nulls across T*N):")
            for j, col in enumerate(self.dyn_cols):
                cnt = int(np.isfinite(dyn[:,:,j]).sum())
                log.info(f"  {col:12s}: {cnt}")

        if fit_scalers:
            self.target_stats = self._compute_target_stats(aqi, pm)
        else:
            self.target_stats = {k: dict(v) for k, v in (target_stats or {}).items()}
            if target_stats is None and verbose:
                log.warning("[Dataset] No target_stats supplied; targets remain unscaled.")

        if self.target_stats:
            aqi = self._standardize_target(aqi, "aqi")
            pm = self._standardize_target(pm, "pm25")

        # scalers
        from src.utils import fit_scalers
        if fit_scalers:
            flat = dyn.reshape(-1, Ddyn)
            mask = ~np.isnan(flat).any(axis=1)
            train_dyn = flat[mask] if mask.any() else np.zeros((1, Ddyn), dtype=np.float32)
            self.dyn_scaler, self.stat_scaler = fit_scalers(train_dyn, stat)
        else:
            self.dyn_scaler, self.stat_scaler = dyn_scaler, stat_scaler

        # scale stat once
        from sklearn.preprocessing import StandardScaler
        if self.stat_scaler is None:
            self.stat_scaler = StandardScaler().fit(stat)
        stat_s = self.stat_scaler.transform(stat)
        self.stat_s = torch.tensor(stat_s, dtype=torch.float32)

        # ventilation mean
        u_idx = self.dyn_cols.index("U") if "U" in self.dyn_cols else None
        p_idx = self.dyn_cols.index("PBLH") if "PBLH" in self.dyn_cols else None
        if u_idx is not None and p_idx is not None:
            VC = dyn[:,:,u_idx] * np.nan_to_num(dyn[:,:,p_idx], nan=0.0)
            self.vc_mean = float(np.nanmean(VC[np.isfinite(VC)])) if np.isfinite(VC).any() else 1.0
        else:
            self.vc_mean = 1.0
        log.info(f"[Dataset] Ventilation mean (train context): {self.vc_mean:.3f}")

        # store
        self.graph = PhysicsGraph(self.nodes, cfg)
        self.dyn = dyn
        self.aqi = aqi
        self.pm  = pm

    def __len__(self): return len(self.valid_t_idx)

    def __getitem__(self, i):
        t_end = self.valid_t_idx[i]
        t0 = t_end - (self.W-1)

        # [W,N,D] -> impute per-feature mean within window; fallback=0 for all-NaN
        x_dyn = self.dyn[t0:t_end+1,:,:]  # [W,N,D]
        W,N,D = x_dyn.shape
        flat = x_dyn.reshape(-1, D)
        m = np.nanmean(flat, axis=0)
        m = np.where(np.isfinite(m), m, 0.0)
        imp = np.where(np.isnan(flat), m, flat)

        if self.dyn_scaler is not None:
            flat_s = self.dyn_scaler.transform(imp)
        else:
            flat_s = imp

        x_dyn_s = flat_s.reshape(W, N, D).transpose(1,0,2)  # [N,W,D]
        x_dyn_s = torch.tensor(x_dyn_s, dtype=torch.float32)
        x_stat  = self.stat_s

        # targets at t_end
        y_aqi = torch.tensor(self.aqi[t_end], dtype=torch.float32)
        y_pm  = torch.tensor(self.pm[t_end], dtype=torch.float32)
        mask_aqi = ~torch.isnan(y_aqi)
        mask_pm  = ~torch.isnan(y_pm)

        # physics edges at t_end
        def idx(c): return self.dyn_cols.index(c) if c in self.dyn_cols else None
        U  = self.dyn[t_end,:,idx("U")]     if idx("U") is not None else np.zeros(self.graph.N, np.float32)
        TH = self.dyn[t_end,:,idx("theta")] if idx("theta") is not None else np.zeros(self.graph.N, np.float32)
        P  = self.dyn[t_end,:,idx("PBLH")]  if idx("PBLH") is not None else np.zeros(self.graph.N, np.float32)
        U  = np.nan_to_num(U, nan=0.0); TH = np.nan_to_num(TH, nan=0.0); P = np.nan_to_num(P, nan=0.0)

        edge_index, edge_weight = self.graph.build(U, TH, P, self.vc_mean)

        d = GData()
        d.x_dyn = x_dyn_s
        d.x_stat = x_stat
        d.edge_index = edge_index
        d.edge_weight = edge_weight
        d.y_aqi = y_aqi
        d.y_pm = y_pm
        d.mask_aqi = mask_aqi
        d.mask_pm = mask_pm
        d.t_end = int(self.times[t_end])
        return d

    @staticmethod
    def _stat_summary(arr):
        flat = arr[np.isfinite(arr)]
        if flat.size == 0:
            return dict(mean=0.0, std=1.0)
        mu = float(flat.mean())
        sd = float(flat.std(ddof=0))
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        return dict(mean=mu, std=sd)

    def _compute_target_stats(self, aqi, pm):
        stats = {"aqi": self._stat_summary(aqi)}
        stats["pm25"] = self._stat_summary(pm)
        return stats

    def _standardize_target(self, arr, key):
        stats = self.target_stats.get(key)
        if not stats:
            return arr
        mu = stats.get("mean", 0.0)
        sd = stats.get("std", 1.0)
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        return (arr - mu) / sd
