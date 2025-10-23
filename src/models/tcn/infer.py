import torch
import pyarrow.parquet as pq
import numpy as np
from .model import TCN

def load_model(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt["cfg"]
    in_ch = ckpt["in_channels"]
    model = TCN(
        in_channels=in_ch,
        hidden=cfg["model"]["hidden_channels"],
        layers=cfg["model"]["layers"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
        use_group_norm=cfg["model"]["use_group_norm"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg

@torch.no_grad()
def predict_last_window(features_path: str, lookback: int, model_path: str):
    model, cfg = load_model(model_path)
    tbl = pq.read_table(features_path)
    df = tbl.to_pandas().sort_values("timestamp")
    feat_cols = [c for c in df.columns if c != "timestamp"]
    x = df.tail(lookback)[feat_cols].to_numpy(dtype=np.float32).T  # F × L
    x = torch.from_numpy(x).unsqueeze(0)
    logits = model(x)
    prob_up = torch.softmax(logits, dim=1)[0, 1].item()
    return {"prob_up": prob_up, "thresholds": cfg["thresholds"]["grid"]}
