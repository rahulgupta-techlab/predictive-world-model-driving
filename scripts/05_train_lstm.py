"""Train a small PyTorch LSTM trajectory forecaster (optional dependency: torch)."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("PyTorch is optional. Install a CUDA-compatible build from https://pytorch.org/get-started/locally/") from exc

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/processed/trajectories.npz")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--output", default="experiments/results/lstm.pt")
    args = p.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    train = data["split"] == "train"
    val = data["split"] == "validation"
    if not train.any() or not val.any():
        raise SystemExit("Training and validation splits are required. Collect at least three episodes.")
    x_train = torch.tensor(data["x"][train], dtype=torch.float32)
    y_train = torch.tensor(data["y"][train], dtype=torch.float32)
    x_val = torch.tensor(data["x"][val], dtype=torch.float32)
    y_val = torch.tensor(data["y"][val], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TrajectoryLSTM(nn.Module):
        def __init__(self, hidden, future_steps):
            super().__init__()
            self.encoder = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                      nn.Linear(hidden, future_steps * 2))
            self.future_steps = future_steps

        def forward(self, inputs):
            _, (hidden, _) = self.encoder(inputs)
            return self.head(hidden[-1]).view(-1, self.future_steps, 2)

    model = TrajectoryLSTM(args.hidden_size, y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    best_val = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val.to(device)), y_val.to(device)).cpu())
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": val_loss}
        history.append(row)
        print(json.dumps(row))
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "obs_steps": int(x_train.shape[1]),
                        "pred_steps": int(y_train.shape[1]),
                        "hidden_size": args.hidden_size,
                        "validation_loss": best_val}, output)

    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint saved to {output}; device={device}; best_validation_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
