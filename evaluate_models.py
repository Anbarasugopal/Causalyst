"""
evaluate_models.py

Comprehensive evaluation for Causalyst's three model variants:
  1. Causal-graph GNN (Granger + FDR edges, use_gcn=True)
  2. No-graph baseline (GRU-only, use_gcn=False)
  3. Correlation-graph baseline (top-K absolute correlations, use_gcn=True)

Tasks implemented:
  Task 1: Precision/recall/F1 threshold sweep on validation, then test eval at tuned threshold
  Task 2: Multi-seed stability check (3 seeds)
  Task 3: Class-imbalance ablation (pos_weight in BCEWithLogitsLoss)

Architecture and data pipeline replicate the Colab notebook exactly:
  - CausalTemporalGNN: GRU(1→16) + 2×GCNConv(16→16) + Linear(32→16→1)
  - Window=20 trading days, labels=(next_day_return > 0)
  - Chronological split: train≤2019, val=2020, test=2021
  - BCEWithLogitsLoss, Adam(lr=1e-3), 20 epochs, batch_size=8

Per PROJECT_HANDOFF.md constraints:
  - Full 376-stock graph runs every inference (constraint #1)
  - Causal graph from cached full-history discovery (constraint #2)
  - No hardcoded metrics (constraint #8)
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_curve
)

# ─── Configuration ──────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).parent / "backend_artifacts-20260810T073325Z-1-001" / "backend_artifacts"
RETURNS_FILE = ARTIFACTS_DIR / "returns.parquet"
CAUSAL_GRAPH_FILE = ARTIFACTS_DIR / "causal_graph.pt"

WINDOW = 20
HIDDEN = 16
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3

TRAIN_END = pd.Timestamp("2019-12-31")
VAL_START = pd.Timestamp("2020-01-01")
VAL_END = pd.Timestamp("2020-12-31")
TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2021-12-31")

THRESHOLD_SWEEP = np.arange(0.30, 0.72, 0.02)  # 0.30 to 0.70 inclusive

SEEDS = [42, 123, 7]  # 3 seeds for stability check

DEVICE = torch.device("cpu")

OUTPUT_DIR = Path(__file__).parent / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Dataset ────────────────────────────────────────────────────────────────
class StockWindowDataset(Dataset):
    """Exactly replicates the Colab notebook's StockWindowDataset."""

    def __init__(self, returns_np, dates, window, start_date=None, end_date=None):
        self.returns = returns_np  # (n_days, n_stocks)
        self.window = window
        self.indices = []
        for t in range(window, len(dates) - 1):
            label_date = dates[t + 1]
            if start_date and label_date < start_date:
                continue
            if end_date and label_date > end_date:
                continue
            self.indices.append(t)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        t = self.indices[item]
        x = self.returns[t - self.window:t, :].T  # (n_stocks, window)
        y = (self.returns[t + 1, :] > 0).astype(np.float32)  # (n_stocks,)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ─── Model ──────────────────────────────────────────────────────────────────
class CausalTemporalGNN(nn.Module):
    """Exact replica of the Colab model architecture."""

    def __init__(self, hidden=HIDDEN, use_gcn=True):
        super().__init__()
        self.hidden = hidden
        self.use_gcn = use_gcn
        self.node_gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.gcn1 = GCNConv(hidden, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self._batched_graph_cache = {}

    def _batch_graph(self, edge_index, edge_weight, batch_size, n_nodes, device):
        key = (batch_size, n_nodes, edge_index.shape[1], device.type, str(device))
        if key in self._batched_graph_cache:
            return self._batched_graph_cache[key]
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        if edge_index.numel() == 0:
            out_index = edge_index
            out_weight = edge_weight
        else:
            offsets = (
                torch.arange(batch_size, device=device, dtype=torch.long)
                .repeat_interleave(edge_index.shape[1]) * n_nodes
            )
            out_index = edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)
            out_weight = edge_weight.repeat(batch_size)
        self._batched_graph_cache[key] = (out_index, out_weight)
        return out_index, out_weight

    def forward(self, x, edge_index, edge_weight):
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        batch_size, n_nodes, window = x.shape
        x_seq = x.reshape(batch_size * n_nodes, window, 1)
        _, h_n = self.node_gru(x_seq)
        self_embed = h_n.squeeze(0)

        if self.use_gcn:
            batched_ei, batched_ew = self._batch_graph(
                edge_index, edge_weight, batch_size, n_nodes, x.device
            )
            g1 = torch.relu(self.gcn1(self_embed, batched_ei, batched_ew))
            g2 = self.gcn2(g1, batched_ei, batched_ew)
        else:
            g2 = torch.zeros_like(self_embed)

        combined = torch.cat([self_embed, g2], dim=-1)
        logits = self.head(combined).squeeze(-1).view(batch_size, n_nodes)
        return logits.squeeze(0) if squeeze else logits


# ─── Data Loading ───────────────────────────────────────────────────────────
def load_data():
    """Load returns, split into train/val/test, create dataloaders."""
    print("Loading returns data...")
    returns_df = pd.read_parquet(RETURNS_FILE)
    tickers = list(returns_df.columns)
    n_stocks = len(tickers)
    print(f"  {returns_df.shape[0]} days × {n_stocks} stocks")
    print(f"  Date range: {returns_df.index.min()} to {returns_df.index.max()}")

    returns_np = returns_df.values.astype(np.float32)
    dates = returns_df.index

    train_ds = StockWindowDataset(returns_np, dates, WINDOW, end_date=TRAIN_END)
    val_ds = StockWindowDataset(returns_np, dates, WINDOW, start_date=VAL_START, end_date=VAL_END)
    test_ds = StockWindowDataset(returns_np, dates, WINDOW, start_date=TEST_START, end_date=TEST_END)

    print(f"  Split sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Compute class balance for train set (needed for pos_weight)
    train_labels = []
    for i in range(len(train_ds)):
        _, y = train_ds[i]
        train_labels.append(y.numpy())
    train_labels = np.concatenate(train_labels)
    pos_rate = train_labels.mean()
    neg_rate = 1 - pos_rate
    pos_weight_value = neg_rate / pos_rate
    print(f"  Train positive rate: {pos_rate:.4f}, pos_weight: {pos_weight_value:.4f}")

    return returns_df, tickers, train_ds, val_ds, test_ds, pos_weight_value


def load_graphs(returns_df, tickers):
    """Load causal graph, build correlation graph (density-matched)."""
    print("\nLoading causal graph...")
    g = torch.load(CAUSAL_GRAPH_FILE, weights_only=False)
    causal_edge_index = g["edge_index"]
    causal_edge_weight = g["edge_weight"]
    n_causal_edges = causal_edge_index.shape[1]
    print(f"  Causal edges: {n_causal_edges}")

    # No-graph: empty edges
    no_edge_index = torch.zeros(2, 0, dtype=torch.long)
    no_edge_weight = torch.zeros(0, dtype=torch.float32)

    # Correlation graph: density-matched to causal graph
    print("Building correlation graph (density-matched)...")
    train_returns = returns_df.loc[:TRAIN_END]
    corr_matrix = train_returns.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)

    # Get top-K correlations matching causal graph density
    n_stocks = len(tickers)
    # Flatten upper triangle to get unique pairs, then take top-K directed edges
    all_edges = []
    for i in range(n_stocks):
        for j in range(n_stocks):
            if i != j:
                all_edges.append((i, j, corr_matrix.iloc[i, j]))
    all_edges.sort(key=lambda x: x[2], reverse=True)
    top_k = all_edges[:n_causal_edges]

    corr_src = [e[0] for e in top_k]
    corr_tgt = [e[1] for e in top_k]
    corr_w = [e[2] for e in top_k]
    corr_edge_index = torch.tensor([corr_src, corr_tgt], dtype=torch.long)
    corr_edge_weight = torch.tensor(corr_w, dtype=torch.float32)
    print(f"  Correlation edges: {corr_edge_index.shape[1]}")

    graphs = {
        "causal_graph": (causal_edge_index, causal_edge_weight, True),
        "no_graph": (no_edge_index, no_edge_weight, False),
        "correlation_graph": (corr_edge_index, corr_edge_weight, True),
    }
    return graphs


# ─── Training ───────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_logits_and_labels(model, loader, edge_index, edge_weight):
    """Collect raw logits and true labels for threshold analysis."""
    model.eval()
    all_logits, all_true = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x, edge_index, edge_weight)
        all_logits.append(logits.cpu().numpy().ravel())
        all_true.append(y.numpy().ravel())
    return np.concatenate(all_logits), np.concatenate(all_true)


def evaluate_at_threshold(logits, labels, threshold):
    """Compute metrics at a specific sigmoid threshold."""
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "predicted_positive_rate": float(preds.mean()),
        "true_positive_rate": float(labels.mean()),
    }


def threshold_sweep(logits, labels, thresholds):
    """Sweep thresholds, return all results and best-F1 threshold."""
    results = []
    for t in thresholds:
        results.append(evaluate_at_threshold(logits, labels, t))
    best = max(results, key=lambda r: r["f1"])
    return results, best["threshold"]


def train_one_model(model_name, edge_index, edge_weight, use_gcn,
                    train_loader, val_loader, test_loader,
                    seed, pos_weight=None, epochs=EPOCHS):
    """Train a single model variant, return val/test logits and labels."""
    set_seed(seed)

    model = CausalTemporalGNN(hidden=HIDDEN, use_gcn=use_gcn).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if pos_weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

    best_val_f1 = -1.0
    best_state = None

    tag = f"{model_name}/seed={seed}" + (f"/pw={pos_weight:.2f}" if pos_weight else "")
    print(f"\n  Training {tag} for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edge_index, edge_weight)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Quick val check at threshold=0.5 for early stopping / best model
        val_logits, val_labels = collect_logits_and_labels(model, val_loader, edge_index, edge_weight)
        val_metrics = evaluate_at_threshold(val_logits, val_labels, 0.5)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 1 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:2d}: train_loss={np.mean(train_losses):.4f}, "
                  f"val_f1={val_metrics['f1']:.4f}, val_prec={val_metrics['precision']:.4f}, "
                  f"val_rec={val_metrics['recall']:.4f}, val_pred_pos={val_metrics['predicted_positive_rate']:.4f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    # Collect final logits
    val_logits, val_labels = collect_logits_and_labels(model, val_loader, edge_index, edge_weight)
    test_logits, test_labels = collect_logits_and_labels(model, test_loader, edge_index, edge_weight)

    return model, val_logits, val_labels, test_logits, test_labels, best_val_f1


# ─── Main Tasks ─────────────────────────────────────────────────────────────
def run_task1(train_ds, val_ds, test_ds, graphs, pos_weight_value):
    """Task 1: Threshold sweep and precision/recall breakdown."""
    print("\n" + "=" * 70)
    print("TASK 1: PRECISION/RECALL BREAKDOWN + THRESHOLD TUNING")
    print("=" * 70)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    results = {}
    for model_name, (edge_index, edge_weight, use_gcn) in graphs.items():
        print(f"\n--- Model: {model_name} ---")
        model, val_logits, val_labels, test_logits, test_labels, best_val_f1 = \
            train_one_model(model_name, edge_index, edge_weight, use_gcn,
                            train_loader, val_loader, test_loader, seed=42)

        # Threshold sweep on validation
        sweep, best_threshold = threshold_sweep(val_logits, val_labels, THRESHOLD_SWEEP)

        # Test evaluation at tuned threshold
        test_metrics = evaluate_at_threshold(test_logits, test_labels, best_threshold)

        # Also report at default 0.5 for comparison
        test_at_05 = evaluate_at_threshold(test_logits, test_labels, 0.5)
        val_at_best = evaluate_at_threshold(val_logits, val_labels, best_threshold)

        results[model_name] = {
            "val_sweep": sweep,
            "best_val_threshold": best_threshold,
            "val_at_tuned_threshold": val_at_best,
            "test_at_tuned_threshold": test_metrics,
            "test_at_default_05": test_at_05,
        }

        print(f"\n  Best validation threshold: {best_threshold:.2f}")
        print(f"  Val  @ {best_threshold:.2f}: P={val_at_best['precision']:.4f}, "
              f"R={val_at_best['recall']:.4f}, F1={val_at_best['f1']:.4f}, "
              f"pred_pos_rate={val_at_best['predicted_positive_rate']:.4f}")
        print(f"  Test @ {best_threshold:.2f}: P={test_metrics['precision']:.4f}, "
              f"R={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}, "
              f"pred_pos_rate={test_metrics['predicted_positive_rate']:.4f}")
        print(f"  Test @ 0.50 (default): P={test_at_05['precision']:.4f}, "
              f"R={test_at_05['recall']:.4f}, F1={test_at_05['f1']:.4f}, "
              f"pred_pos_rate={test_at_05['predicted_positive_rate']:.4f}")

    return results


def run_task2(train_ds, val_ds, test_ds, graphs, task1_thresholds):
    """Task 2: Multi-seed stability check."""
    print("\n" + "=" * 70)
    print("TASK 2: MULTI-SEED STABILITY CHECK")
    print("=" * 70)

    results = {}
    for model_name, (edge_index, edge_weight, use_gcn) in graphs.items():
        print(f"\n--- Model: {model_name} ---")
        seed_results = []
        tuned_threshold = task1_thresholds[model_name]

        for seed in SEEDS:
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      generator=torch.Generator().manual_seed(seed))
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            model, val_logits, val_labels, test_logits, test_labels, best_val_f1 = \
                train_one_model(model_name, edge_index, edge_weight, use_gcn,
                                train_loader, val_loader, test_loader, seed=seed)

            # Use per-seed validation to find this seed's best threshold
            _, seed_threshold = threshold_sweep(val_logits, val_labels, THRESHOLD_SWEEP)
            test_metrics = evaluate_at_threshold(test_logits, test_labels, seed_threshold)

            seed_results.append({
                "seed": seed,
                "tuned_threshold": seed_threshold,
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_pred_pos_rate": test_metrics["predicted_positive_rate"],
            })
            print(f"  Seed {seed}: threshold={seed_threshold:.2f}, "
                  f"test_F1={test_metrics['f1']:.4f}, P={test_metrics['precision']:.4f}, "
                  f"R={test_metrics['recall']:.4f}, pred_pos={test_metrics['predicted_positive_rate']:.4f}")

        f1s = [r["test_f1"] for r in seed_results]
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        spread = max(f1s) - min(f1s)

        results[model_name] = {
            "seeds": seed_results,
            "mean_test_f1": float(mean_f1),
            "std_test_f1": float(std_f1),
            "spread_test_f1": float(spread),
        }
        print(f"  Summary: F1 = {mean_f1:.4f} ± {std_f1:.4f} (spread={spread:.4f})")

    return results


def run_task3(train_ds, val_ds, test_ds, graphs, pos_weight_value, task1_results):
    """Task 3: Class-imbalance ablation with pos_weight for baselines."""
    print("\n" + "=" * 70)
    print("TASK 3: CLASS-IMBALANCE ABLATION (pos_weight)")
    print("=" * 70)
    print(f"  Using pos_weight = {pos_weight_value:.4f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    results = {}
    # Only retrain the two baselines with pos_weight
    for model_name in ["no_graph", "correlation_graph"]:
        edge_index, edge_weight, use_gcn = graphs[model_name]
        print(f"\n--- Model: {model_name} + pos_weight ---")

        model, val_logits, val_labels, test_logits, test_labels, best_val_f1 = \
            train_one_model(model_name, edge_index, edge_weight, use_gcn,
                            train_loader, val_loader, test_loader,
                            seed=42, pos_weight=pos_weight_value)

        # Threshold sweep
        sweep, best_threshold = threshold_sweep(val_logits, val_labels, THRESHOLD_SWEEP)
        test_metrics = evaluate_at_threshold(test_logits, test_labels, best_threshold)

        # Compare to Task 1 results (without pos_weight)
        t1_test = task1_results[model_name]["test_at_tuned_threshold"]

        results[model_name] = {
            "without_pos_weight": t1_test,
            "with_pos_weight": {
                "best_val_threshold": best_threshold,
                "test_metrics": test_metrics,
                "val_sweep": sweep,
            },
        }
        print(f"\n  WITHOUT pos_weight: threshold={t1_test['threshold']:.2f}, "
              f"P={t1_test['precision']:.4f}, R={t1_test['recall']:.4f}, "
              f"F1={t1_test['f1']:.4f}, pred_pos={t1_test['predicted_positive_rate']:.4f}")
        print(f"  WITH    pos_weight: threshold={best_threshold:.2f}, "
              f"P={test_metrics['precision']:.4f}, R={test_metrics['recall']:.4f}, "
              f"F1={test_metrics['f1']:.4f}, pred_pos={test_metrics['predicted_positive_rate']:.4f}")

        if test_metrics["f1"] > 0.05 and t1_test["f1"] < 0.05:
            print(f"  *** pos_weight FIXES collapse for {model_name}! ***")
        elif test_metrics["f1"] < 0.05:
            print(f"  pos_weight did NOT fix collapse for {model_name}")

    return results


def main():
    start_time = time.time()

    # Load data
    returns_df, tickers, train_ds, val_ds, test_ds, pos_weight_value = load_data()
    graphs = load_graphs(returns_df, tickers)

    # Task 1
    task1_results = run_task1(train_ds, val_ds, test_ds, graphs, pos_weight_value)

    # Extract tuned thresholds for Task 2
    task1_thresholds = {
        name: res["best_val_threshold"] for name, res in task1_results.items()
    }

    # Task 2
    task2_results = run_task2(train_ds, val_ds, test_ds, graphs, task1_thresholds)

    # Task 3
    task3_results = run_task3(train_ds, val_ds, test_ds, graphs, pos_weight_value, task1_results)

    # Save all results
    all_results = {
        "task1_threshold_tuning": task1_results,
        "task2_multi_seed": task2_results,
        "task3_pos_weight_ablation": task3_results,
        "config": {
            "window": WINDOW,
            "hidden": HIDDEN,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seeds": SEEDS,
            "n_stocks": len(tickers),
            "threshold_sweep_range": [float(THRESHOLD_SWEEP[0]), float(THRESHOLD_SWEEP[-1])],
            "pos_weight": pos_weight_value,
        },
        "elapsed_seconds": time.time() - start_time,
    }

    # Custom serializer for numpy types
    def np_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    output_file = OUTPUT_DIR / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=np_serializer)

    # Also save metadata.json at project root for backend API (constraint #8)
    metadata = {
        "n_stocks": len(tickers),
        "tickers": tickers,
        "tuned_thresholds": task1_thresholds,
        "task1_test_metrics": {
            name: res["test_at_tuned_threshold"] for name, res in task1_results.items()
        },
        "task2_stability": {
            name: {
                "mean_f1": res["mean_test_f1"],
                "std_f1": res["std_test_f1"],
                "spread_f1": res["spread_test_f1"],
            }
            for name, res in task2_results.items()
        },
        "task3_pos_weight_effect": {
            name: {
                "f1_without": res["without_pos_weight"]["f1"],
                "f1_with": res["with_pos_weight"]["test_metrics"]["f1"],
            }
            for name, res in task3_results.items()
        },
        "pos_weight_value": pos_weight_value,
        "window": WINDOW,
        "hidden": HIDDEN,
    }
    metadata_file = Path(__file__).parent / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=np_serializer)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"ALL TASKS COMPLETE in {elapsed:.1f}s")
    print(f"Full results: {output_file}")
    print(f"Metadata: {metadata_file}")
    print(f"{'=' * 70}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<25} {'Thresh':>6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'PredPos':>8}")
    print("-" * 70)
    for name in ["causal_graph", "no_graph", "correlation_graph"]:
        t1 = task1_results[name]["test_at_tuned_threshold"]
        print(f"{name:<25} {t1['threshold']:>6.2f} {t1['precision']:>7.4f} "
              f"{t1['recall']:>7.4f} {t1['f1']:>7.4f} {t1['predicted_positive_rate']:>8.4f}")

    print(f"\n{'Model':<25} {'Mean F1':>8} {'Std':>7} {'Spread':>7}")
    print("-" * 50)
    for name in ["causal_graph", "no_graph", "correlation_graph"]:
        t2 = task2_results[name]
        print(f"{name:<25} {t2['mean_test_f1']:>8.4f} {t2['std_test_f1']:>7.4f} "
              f"{t2['spread_test_f1']:>7.4f}")

    print(f"\npos_weight ablation (baselines only):")
    print(f"{'Model':<25} {'F1 w/o':>8} {'F1 w/':>8} {'Delta':>8}")
    print("-" * 50)
    for name in ["no_graph", "correlation_graph"]:
        t3 = task3_results[name]
        f1_without = t3["without_pos_weight"]["f1"]
        f1_with = t3["with_pos_weight"]["test_metrics"]["f1"]
        delta = f1_with - f1_without
        print(f"{name:<25} {f1_without:>8.4f} {f1_with:>8.4f} {delta:>+8.4f}")


if __name__ == "__main__":
    main()
