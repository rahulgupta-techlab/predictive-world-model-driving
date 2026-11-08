"""Train a Transformer encoder trajectory forecaster.

Input/output arrays follow the actor-centric local-frame convention produced by
03_prepare_trajectory_dataset.py. This is a research baseline, not a driving policy.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("Install a CUDA-compatible PyTorch build from https://pytorch.org/get-started/locally/") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/trajectories.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/results/transformer.pt")
    args = parser.parse_args()

    if args.d_model % args.heads:
        parser.error("--d-model must be divisible by --heads")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = np.load(args.dataset, allow_pickle=False)
    train_mask, val_mask = data["split"] == "train", data["split"] == "validation"
    if not train_mask.any() or not val_mask.any():
        raise SystemExit("Need non-empty train and validation splits. Collect at least three distinct episodes.")
    x_train = torch.tensor(data["x"][train_mask], dtype=torch.float32)
    y_train = torch.tensor(data["y"][train_mask], dtype=torch.float32)
    x_val = torch.tensor(data["x"][val_mask], dtype=torch.float32)
    y_val = torch.tensor(data["y"][val_mask], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class PositionalEncoding(nn.Module):
        def __init__(self, width, max_len=512):
            super().__init__()
            pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
            pe = torch.zeros(max_len, width)
            pe[:, 0::2] = torch.sin(pos * div)
            if width > 1:
                pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.shape[1]]

    class TrajectoryTransformer(nn.Module):
        def __init__(self, width, heads, layers, dropout, future_steps):
            super().__init__()
            self.input_projection = nn.Linear(2, width)
            self.position = PositionalEncoding(width)
            layer = nn.TransformerEncoderLayer(
                d_model=width, nhead=heads, dim_feedforward=width * 4,
                dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                                  enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                      nn.Linear(width, future_steps * 2))
            self.future_steps = future_steps

        def forward(self, inputs):
            encoded = self.encoder(self.position(self.input_projection(inputs)))
            context = self.norm(encoded[:, -1])
            return self.head(context).view(-1, self.future_steps, 2)

    model = TrajectoryTransformer(args.d_model, args.heads, args.layers,
                                  args.dropout, y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(x_train, y_train),
                        batch_size=args.batch_size, shuffle=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_val, stale = float("inf"), 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val.to(device)), y_val.to(device)).cpu())
        scheduler.step(val_loss)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "validation_loss": val_loss,
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(json.dumps(record))
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "obs_steps": int(x_train.shape[1]),
                "pred_steps": int(y_train.shape[1]),
                "d_model": args.d_model, "heads": args.heads,
                "layers": args.layers, "dropout": args.dropout,
                "validation_loss": best_val, "seed": args.seed
            }, output)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {output}; device={device}; validation_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
