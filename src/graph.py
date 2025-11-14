import numpy as np
import torch
from src.utils import get_logger

log = get_logger(__name__)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def bearing_math_rad(lat1, lon1, lat2, lon2):
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r)*np.sin(lat2r) - np.sin(lat1r)*np.cos(lat2r)*np.cos(dlon)
    brng_from_north = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    theta_math_deg = (90.0 - brng_from_north) % 360.0  # 0=east, CCW+
    return np.deg2rad(theta_math_deg)

class PhysicsGraph:
    def __init__(self, nodes_df, cfg):
        self.nodes = nodes_df.reset_index(drop=True)
        self.N = len(self.nodes)
        self.k = cfg["physics"]["knn_k"]
        self.ell = cfg["physics"]["dist_length_scale_m"]
        self.gamma = cfg["physics"]["wind_align_gamma"]
        self.beta = cfg["physics"]["beta_weight"]
        self.row_norm = cfg["physics"]["row_normalize"]
        self.use_wind = cfg["physics"]["use_wind"]
        self.use_pblh = cfg["physics"]["use_pblh"]

        self.lat = self.nodes["lat"].values
        self.lon = self.nodes["lon"].values

        log.info(f"[Graph] Building base kNN graph: N={self.N}, k={self.k}, ell={self.ell} m")
        D = np.zeros((self.N, self.N), dtype=np.float32)
        for i in range(self.N):
            D[i,:] = haversine_m(self.lat[i], self.lon[i], self.lat, self.lon)
            D[i,i] = np.inf
        self.knn_idx = np.argsort(D, axis=1)[:, :self.k]
        self.base_w = np.exp(-D / max(1e-6, self.ell)).astype(np.float32)

        self.bearing = np.zeros((self.N, self.k), dtype=np.float32)
        for i in range(self.N):
            nbrs = self.knn_idx[i]
            self.bearing[i,:] = bearing_math_rad(self.lat[i], self.lon[i], self.lat[nbrs], self.lon[nbrs])

    def build(self, U_vec, theta_vec, PBLH_vec, vc_mean=1.0):
        N, K = self.N, self.k
        rows = np.repeat(np.arange(N), K)
        cols = self.knn_idx.reshape(-1)
        w = self.base_w[rows, cols].copy()

        if self.use_wind:
            theta_src = theta_vec[rows]
            bearing_ij = self.bearing.reshape(-1)
            g = np.clip(np.cos(theta_src - bearing_ij), 0.0, 1.0) ** self.gamma
        else:
            g = 0.0

        if self.use_pblh:
            VC = U_vec * np.maximum(PBLH_vec, 0.0)
            vc_src = (VC / (vc_mean + 1e-6))[rows]
        else:
            vc_src = 0.0

        if self.use_wind or self.use_pblh:
            w = w * (1.0 + self.beta * (g * vc_src))

        w = np.clip(w, 0.0, None)

        if self.row_norm:
            s = np.zeros(N, dtype=np.float32)
            for i in range(N):
                s[i] = w[i*K:(i+1)*K].sum()
            s[s==0.0] = 1.0
            w = w / np.repeat(s, K)

        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
        edge_weight = torch.tensor(w, dtype=torch.float32)
        return edge_index, edge_weight
