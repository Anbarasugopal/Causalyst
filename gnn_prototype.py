"""
gnn_prototype.py

Validates the RECOMMENDED architecture (simple GRU-per-node encoder +
GCN graph propagation, built from base torch_geometric layers rather than
a specialized temporal-GNN library) actually runs end-to-end, before
committing to building it at 500-stock scale on Colab.

This is deliberately the SIMPLEST version that could plausibly work --
the point of this prototype is to catch shape/API mistakes cheaply on 10
real stocks and a few CPU-seconds, not to produce a good model. Accuracy
here is meaningless (10 stocks, few epochs, no real validation split).
"""
import sys
sys.path.insert(0, "/home/claude/gnn_proto")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from causal_discovery import granger_edges

DATA_DIR = Path("/home/claude/gnn_proto/data/nse_real")
STOCK_FILES = {
    "RELIANCE": "000_RELIANCE.csv", "TCS": "001_TCS.csv", "HDFCBANK": "003_HDFCBANK.csv",
    "INFY": "005_INFY.csv", "KOTAKBANK": "006_KOTAKBANK.csv", "ICICIBANK": "009_ICICIBANK.csv",
    "SBIN": "010_SBIN.csv", "HCLTECH": "015_HCLTECH.csv", "WIPRO": "017_WIPRO.csv",
    "AXISBANK": "018_AXISBANK.csv",
}

WINDOW = 20  # days of history fed to the per-node GRU


def load_returns():
    series = {}
    for ticker, fname in STOCK_FILES.items():
        df = pd.read_csv(DATA_DIR / fname, parse_dates=["Date"]).sort_values("Date").set_index("Date")
        series[ticker] = df["Close"]
    prices = pd.DataFrame(series).dropna()
    return np.log(prices / prices.shift(1)).dropna()


def build_edge_index_and_weight(edges, tickers):
    idx = {t: i for i, t in enumerate(tickers)}
    src, tgt, w = [], [], []
    for (a, b), info in edges.items():
        src.append(idx[a]); tgt.append(idx[b])
        w.append(-np.log10(max(info["p_value"], 1e-12)))
    if not src:
        src, tgt, w = list(range(len(tickers))), list(range(len(tickers))), [1.0] * len(tickers)
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
    return edge_index, edge_weight


def make_windows(returns_df, window=WINDOW):
    arr = returns_df.values
    n_days, n_stocks = arr.shape
    X, y = [], []
    for t in range(window, n_days - 1):
        X.append(arr[t-window:t, :].T)
        y.append((arr[t+1, :] > 0).astype(np.float32))
    return np.stack(X), np.stack(y)


class CausalTemporalGNN(nn.Module):
    def __init__(self, window=WINDOW, hidden=16):
        super().__init__()
        self.node_gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.gcn1 = GCNConv(hidden, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_weight):
        x_seq = x.unsqueeze(-1)
        _, h_n = self.node_gru(x_seq)
        self_embed = h_n.squeeze(0)

        g1 = torch.relu(self.gcn1(self_embed, edge_index, edge_weight))
        g2 = self.gcn2(g1, edge_index, edge_weight)

        combined = torch.cat([self_embed, g2], dim=-1)
        logits = self.head(combined).squeeze(-1)
        return logits


if __name__ == "__main__":
    returns = load_returns()
    tickers = list(returns.columns)
    print(f"Loaded {len(returns)} real trading days, {len(tickers)} stocks")

    print("\nRunning causal discovery (this is the part that does NOT scale trivially to 500 stocks)...")
    edges = granger_edges(returns, tickers, max_lag=2, alpha=0.05, bonferroni=False)
    print(f"Found {len(edges)} causal edges")
    edge_index, edge_weight = build_edge_index_and_weight(edges, tickers)
    print(f"edge_index shape: {edge_index.shape}, edge_weight shape: {edge_weight.shape}")

    print("\nBuilding training windows...")
    X, y = make_windows(returns)
    print(f"X shape: {X.shape}  (n_samples, n_stocks, window)")
    print(f"y shape: {y.shape}  (n_samples, n_stocks)")

    X_t = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)

    model = CausalTemporalGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    print("\nRunning 5 training steps on the first 5 samples (validation only, not real training)...")
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        logits = model(X_t[step], edge_index, edge_weight)
        loss = loss_fn(logits, y_t[step])
        loss.backward()
        optimizer.step()
        print(f"  step {step}: loss = {loss.item():.4f}")

    print("\nArchitecture validated end-to-end: data pipeline -> GRU node encoder -> "
          "GCN graph propagation -> prediction head -> backprop, all shapes check out.")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
