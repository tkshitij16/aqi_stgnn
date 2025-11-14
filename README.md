# AQI-STT-GNN (Physics-aware, Epoch-only, Locked CSV)

Physics-aware spatiotemporal GNN for **AQI** at Chicago street-scale.

**Locked CSV schema (no changes required):**

**Notes**
- `epoch`: UTC seconds (int64)
- `theta`: **math radians** (0=east, CCW+). Your sample already matches this.
- `PBLH`: meters (optional; set `physics.use_pblh=false` if missing)

## Quickstart

```bash
# 1) Install
pip install -r requirements.txt

# 2) Put your locked master here
# aqi_stgnn/data/stt_master_locked_2024.csv

# 3) Train
python -m src.train --config configs/chicago_aqi_2024.yaml

# 4) Evaluate (tables/metrics.csv)
python -m src.eval --config configs/chicago_aqi_2024.yaml

# 5) Predict full 2024 (outputs/preds_2024*.csv)
python -m src.predict --config configs/chicago_aqi_2024.yaml

# 6) Figures & monthly stats
python -m src.plotting --config configs/chicago_aqi_2024.yaml
