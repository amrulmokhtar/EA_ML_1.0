from typing import Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """
    Builds rolling sequences of length lookback, then reads the label at t + horizon - 1.
    Expects features parquet to have a timestamp column and numeric feature columns.
    Expects labels parquet to have timestamp and label columns, where label in {-1, 0, 1}.
    """
    def __init__(
        self,
        features_path: str,
        labels_path: str,
        lookback: int,
        horizon: int,
        start_time: Optional[str],
        end_time: Optional[str],
        use_neutral: bool = True,
    ):
        feats = pq.read_table(features_path).to_pandas()
        labs = pq.read_table(labels_path).to_pandas()

        feats = feats.sort_values("timestamp").reset_index(drop=True)
        labs = labs.sort_values("timestamp").reset_index(drop=True)

        if start_time is not None:
            feats = feats[feats["timestamp"] >= start_time]
            labs = labs[labs["timestamp"] >= start_time]
        if end_time is not None:
            feats = feats[feats["timestamp"] <= end_time]
            labs = labs[labs["timestamp"] <= end_time]

        df = feats.merge(labs[["timestamp", "label"]], on="timestamp", how="inner")
        if not use_neutral:
            df = df[df["label"] != 0].reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        self.lookback = lookback
        self.horizon = horizon
        self.feature_cols = [c for c in feats.columns if c != "timestamp"]

    def __len__(self) -> int:
        return max(0, len(self.df) - self.lookback - self.horizon + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx
        e = idx + self.lookback
        y_idx = e + self.horizon - 1

        x_np = self.df.loc[s:e - 1, self.feature_cols].to_numpy(dtype=np.float32)  # L by F
        y_val = self.df.loc[y_idx, "label"]
        y_bin = 1 if y_val == 1 else 0  # neutrals are filtered above if needed

        x = torch.from_numpy(x_np).transpose(0, 1)  # F by L for Conv1d
        y = torch.tensor(y_bin, dtype=torch.long)
        return x, y
