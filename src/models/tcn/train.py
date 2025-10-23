import os, argparse, yaml
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import pyarrow.parquet as pq

from .dataset import SequenceDataset
from .model import TCN

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_splits(cfg):
    L = cfg["lookback_bars"]; H = cfg["horizon_bars"]
    p = cfg["data"]; s = cfg["split"]

    ds_train = SequenceDataset(p["features_path"], p["labels_path"], L, H, None, s["train_end"], use_neutral=True)
    ds_val   = SequenceDataset(p["features_path"], p["labels_path"], L, H, s["train_end"], s["val_end"], use_neutral=True)
    ds_test  = SequenceDataset(p["features_path"], p["labels_path"], L, H, s["val_end"], s["test_end"], use_neutral=True)
    return ds_train, ds_val, ds_test

def get_in_channels(features_path: str) -> int:
    cols = pq.read_table(features_path).column_names
    return len([c for c in cols if c != "timestamp"])

def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    n, loss_sum, correct = 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    n, loss_sum, correct = 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tcn_m5.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = get_in_channels(cfg["data"]["features_path"])

    ds_train, ds_val, ds_test = make_splits(cfg)
    train_loader = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    val_loader   = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"], shuffle=False)
    test_loader  = DataLoader(ds_test, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = TCN(
        in_channels=in_ch,
        hidden=cfg["model"]["hidden_channels"],
        layers=cfg["model"]["layers"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
        use_group_norm=cfg["model"]["use_group_norm"],
    ).to(device)

    if cfg["train"]["class_weighting"] == "auto":
        # estimate class weights from a small sample
        import numpy as np
        ys = []
        for i, (_, y) in enumerate(train_loader):
            ys.append(y.numpy())
            if i >= 20:
                break
        y_all = np.concatenate(ys)
        p1 = max(1e-6, (y_all == 1).mean())
        weights = torch.tensor([p1, 1 - p1], dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=1.0 / weights)
    else:
        crit = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["model"]["weight_decay"])

    best_val, patience, waits = -1.0, cfg["train"]["early_stop_patience"], 0
    best_path = cfg["artifacts"]["model_path"]
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    for epoch in range(cfg["train"]["max_epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, crit, opt, device)
        va_loss, va_acc = evaluate(model, val_loader, crit, device)
        print(f"epoch {epoch+1}  train_loss {tr_loss:.4f}  train_acc {tr_acc:.3f}  val_loss {va_loss:.4f}  val_acc {va_acc:.3f}")
        if va_acc > best_val:
            best_val, waits = va_acc, 0
            torch.save({"state_dict": model.state_dict(), "in_channels": in_ch, "cfg": cfg}, best_path)
        else:
            waits += 1
            if waits >= patience:
                print("early stop")
                break

    te_loss, te_acc = evaluate(model, test_loader, crit, device)
    print(f"test_loss {te_loss:.4f}  test_acc {te_acc:.3f}")

if __name__ == "__main__":
    main()
