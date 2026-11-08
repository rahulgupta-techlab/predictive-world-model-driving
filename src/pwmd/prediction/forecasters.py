"""Checkpoint loading and inference for the repository's LSTM/Transformer baselines.

PyTorch is imported lazily so data preparation and unit tests do not require it.
"""
import math
from pathlib import Path
import numpy as np


def load_forecaster(checkpoint_path, device=None):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for learned trajectory inference") from exc
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if "hidden_size" in checkpoint:
        class Model(nn.Module):
            def __init__(self, hidden, future):
                super().__init__()
                self.encoder = nn.LSTM(2, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                          nn.Linear(hidden, future * 2))
                self.future = future
            def forward(self, x):
                _, (h, _) = self.encoder(x)
                return self.head(h[-1]).view(-1, self.future, 2)
        model = Model(checkpoint["hidden_size"], checkpoint["pred_steps"])
        kind = "lstm"
    elif "d_model" in checkpoint:
        class PositionalEncoding(nn.Module):
            def __init__(self, width):
                super().__init__()
                pos = torch.arange(512, dtype=torch.float32).unsqueeze(1)
                div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) *
                                (-math.log(10000.0) / width))
                pe = torch.zeros(512, width)
                pe[:, 0::2] = torch.sin(pos * div)
                if width > 1:
                    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
                self.register_buffer("pe", pe.unsqueeze(0))
            def forward(self, x):
                return x + self.pe[:, :x.shape[1]]
        class Model(nn.Module):
            def __init__(self, width, heads, layers, dropout, future):
                super().__init__()
                self.proj = nn.Linear(2, width)
                self.position = PositionalEncoding(width)
                layer = nn.TransformerEncoderLayer(
                    d_model=width, nhead=heads, dim_feedforward=width * 4,
                    dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                                      enable_nested_tensor=False)
                self.norm = nn.LayerNorm(width)
                self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                          nn.Linear(width, future * 2))
                self.future = future
            def forward(self, x):
                z = self.encoder(self.position(self.proj(x)))
                return self.head(self.norm(z[:, -1])).view(-1, self.future, 2)
        model = Model(checkpoint["d_model"], checkpoint["heads"], checkpoint["layers"],
                      checkpoint["dropout"], checkpoint["pred_steps"])
        kind = "transformer"
    else:
        raise ValueError("Unrecognized checkpoint metadata; expected LSTM or Transformer")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, device, {"model": kind, "obs_steps": int(checkpoint["obs_steps"]),
                           "pred_steps": int(checkpoint["pred_steps"])}


def predict(model, device, observed_xy, batch_size=512):
    """Predict [N, future_steps, 2] from [N, observed_steps, 2] coordinates."""
    import torch
    values = np.asarray(observed_xy, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("observed_xy must have shape [samples, observed_steps, 2]")
    chunks = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start:start + batch_size], device=device)
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, 0, 2), np.float32)
