import json
import math
from datetime import date, timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from torch_geometric.nn import GCNConv

from causalyst.src.agents import build_live_evidence_for_ticker
from causalyst.src.graph_builder import build_sector_lookup_for_tickers

import upstox_auth
import upstox_fetch


app = FastAPI(title="Causalyst Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "backend_artifacts-20260810T073325Z-1-001" / "backend_artifacts"
RETURNS_FILE = ARTIFACTS_DIR / "returns.parquet"
CAUSAL_GRAPH_FILE = ARTIFACTS_DIR / "causal_graph.pt"
CHECKPOINT_FILE = ARTIFACTS_DIR / "model_checkpoint_graph.pt"
METADATA_FILE = ROOT / "metadata.json"
FRONTEND_FILE = ROOT / "stock_prediction_web.html"

WINDOW = 20
HIDDEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CausalTemporalGNN(nn.Module):
    def __init__(self, hidden=HIDDEN, use_gcn=True):
        super().__init__()
        self.hidden = hidden
        self.use_gcn = use_gcn
        self.node_gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.gcn1 = GCNConv(hidden, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
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
                torch.arange(batch_size, device=device, dtype=torch.long).repeat_interleave(edge_index.shape[1])
                * n_nodes
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
            batched_edge_index, batched_edge_weight = self._batch_graph(
                edge_index, edge_weight, batch_size, n_nodes, x.device
            )
            g1 = torch.relu(self.gcn1(self_embed, batched_edge_index, batched_edge_weight))
            g2 = self.gcn2(g1, batched_edge_index, batched_edge_weight)
        else:
            g2 = torch.zeros_like(self_embed)

        combined = torch.cat([self_embed, g2], dim=-1)
        logits = self.head(combined).squeeze(-1).view(batch_size, n_nodes)
        return logits.squeeze(0) if squeeze else logits


model = None
edge_index = None
edge_weight = None
metadata = None
model_metrics = None
base_state = None
returns_df = None
tickers = []
tuned_threshold = 0.5
clusters = []
cluster_by_ticker = {}
cluster_modularity = 0.0
edge_count = 0
last_available_date = None
sector_lookup = {}
sector_coverage = {"official_nse_index_csv": 0, "yahoo_fallback": 0, "missing": 0}


def required_path(path: Path) -> Path:
    if not path.exists():
        raise RuntimeError(f"Required artifact is missing: {path}")
    return path


def build_clusters(graph_payload):
    global clusters, cluster_by_ticker, cluster_modularity, edge_count

    edge_count = int(graph_payload["edge_index"].shape[1])
    edge_records = graph_payload.get("edge_records")
    if edge_records is None:
        ei = graph_payload["edge_index"].detach().cpu().numpy()
        ew = graph_payload["edge_weight"].detach().cpu().numpy()
        ticker_list = graph_payload["ticker_list"]
        edge_records = [
            {
                "source": ticker_list[int(src)],
                "target": ticker_list[int(tgt)],
                "weight": float(weight),
            }
            for src, tgt, weight in zip(ei[0], ei[1], ew)
        ]

    g = nx.Graph()
    g.add_nodes_from(tickers)
    for row in edge_records:
        src = row.get("source") or row.get("from")
        tgt = row.get("target") or row.get("to")
        if src not in tickers or tgt not in tickers:
            continue
        if row.get("p_value") is not None and pd.notna(row.get("p_value")):
            weight = -math.log10(max(float(row["p_value"]), 1e-12))
        else:
            weight = float(row.get("weight", 1.0))
        if g.has_edge(src, tgt):
            g[src][tgt]["weight"] += weight
        else:
            g.add_edge(src, tgt, weight=weight)

    if g.number_of_edges() == 0:
        communities = [{ticker} for ticker in tickers]
        cluster_modularity = 0.0
    else:
        communities = nx.community.louvain_communities(g, weight="weight", seed=42)
        cluster_modularity = float(nx.community.modularity(g, communities, weight="weight"))

    clusters = [sorted(list(c)) for c in sorted(communities, key=lambda c: (-len(c), sorted(c)))]
    cluster_by_ticker = {
        ticker: cluster_id
        for cluster_id, members in enumerate(clusters, start=1)
        for ticker in members
    }


def load_cached_state():
    global returns_df, last_available_date

    returns_df = pd.read_parquet(required_path(RETURNS_FILE))
    missing = [ticker for ticker in tickers if ticker not in returns_df.columns]
    if missing:
        raise RuntimeError(f"returns.parquet is missing {len(missing)} trained tickers, including {missing[:5]}")

    returns_df = returns_df[tickers].sort_index()
    last_available_date = str(pd.Timestamp(returns_df.index.max()).date())
    window = returns_df.tail(WINDOW).values.T
    if window.shape != (len(tickers), WINDOW):
        raise RuntimeError(f"Expected latest window {(len(tickers), WINDOW)}, got {window.shape}")
    return torch.tensor(window, dtype=torch.float32, device=DEVICE)


def try_live_full_graph_state():
    """True live graph inference requires recent returns for every trained node."""
    mapped = getattr(upstox_fetch, "INSTRUMENT_KEYS", {})
    missing = [ticker for ticker in tickers if ticker not in mapped]
    if missing:
        return None, f"Upstox live full-graph window unavailable: mappings cover {len(mapped)} of {len(tickers)} trained stocks"

    token = upstox_auth.load_saved_token()
    if not token:
        return None, "Upstox live full-graph window unavailable: no token; run upstox_auth.py"

    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=60)).isoformat()
    frames = {}
    try:
        for ticker in tickers:
            candles = upstox_fetch.fetch_daily_candles(mapped[ticker], from_date, to_date, token)
            close = pd.to_numeric(candles["close"], errors="coerce")
            returns = np.log(close / close.shift(1)).dropna().tail(WINDOW)
            if len(returns) < WINDOW:
                return None, f"Upstox live full-graph window unavailable: {ticker} returned only {len(returns)} returns"
            frames[ticker] = returns.reset_index(drop=True)
    except Exception as exc:
        return None, f"Upstox live full-graph window unavailable: {exc}"

    live = pd.DataFrame(frames)[tickers]
    return torch.tensor(live.values.T, dtype=torch.float32, device=DEVICE), f"Live Upstox full-graph window through {to_date}"


def cluster_payload(ticker):
    cluster_id = cluster_by_ticker.get(ticker)
    if not cluster_id:
        return None
    members = clusters[cluster_id - 1]
    return {
        "id": cluster_id,
        "size": len(members),
        "peers": [member for member in members if member != ticker][:12],
        "modularity": cluster_modularity,
    }


def causal_model_metrics():
    tuned = metadata.get("task1_test_metrics", {}).get("causal_graph", {})
    return {
        "threshold_tuned": tuned,
        "fixed_threshold": {
            "threshold": 0.5,
            "accuracy": 0.5118501201098147,
            "precision": 0.5002883090673201,
            "recall": 0.30486439002964755,
            "f1": 0.3788600357518115,
        },
        "caveat": (
            "Tuned-threshold F1 was achieved by predicting every test label positive; "
            "fixed-threshold F1 is the more useful evidence that the causal graph model avoided baseline collapse."
        ),
    }


@app.on_event("startup")
def load_assets():
    global model, edge_index, edge_weight, metadata, model_metrics, base_state, tickers, tuned_threshold, sector_lookup, sector_coverage

    with open(required_path(METADATA_FILE), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    tickers = metadata["tickers"]
    sector_lookup = build_sector_lookup_for_tickers(tickers)
    sector_coverage = {"official_nse_index_csv": 0, "missing": 0}
    for ticker in tickers:
        source = sector_lookup.get(ticker.upper(), {}).get("source", "missing")
        sector_coverage[source] = sector_coverage.get(source, 0) + 1
    tuned_threshold = float(metadata["tuned_thresholds"]["causal_graph"])
    model_metrics = causal_model_metrics()

    graph_payload = torch.load(required_path(CAUSAL_GRAPH_FILE), weights_only=False, map_location=DEVICE)
    edge_index = graph_payload["edge_index"].to(DEVICE)
    edge_weight = graph_payload["edge_weight"].to(DEVICE)
    build_clusters(graph_payload)

    checkpoint = torch.load(required_path(CHECKPOINT_FILE), weights_only=False, map_location=DEVICE)
    model = CausalTemporalGNN()
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    base_state = load_cached_state()


@app.get("/")
def index():
    return FileResponse(required_path(FRONTEND_FILE))


@app.get("/api/status")
def status():
    mapped = getattr(upstox_fetch, "INSTRUMENT_KEYS", {})
    return {
        "status": "ready",
        "device": str(DEVICE),
        "n_stocks": len(tickers),
        "window": WINDOW,
        "edge_count": edge_count,
        "cluster_count": len(clusters),
        "cluster_modularity": cluster_modularity,
        "last_available_date": last_available_date,
        "upstox_mapped_tickers": len(mapped),
        "live_full_graph_supported": all(ticker in mapped for ticker in tickers),
        "metrics": model_metrics,
        "tickers": tickers,
        "sector_coverage": sector_coverage,
    }


@app.get("/api/predict")
def predict(ticker: str, days: int = 1, use_live: bool = True):
    ticker = ticker.upper().strip()
    if ticker not in tickers:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not in trained universe.")
    if days < 1 or days > 120:
        raise HTTPException(status_code=400, detail="days must be between 1 and 120 trading days.")

    current_x = base_state.clone()
    data_source = "cached_returns"
    data_message = f"Using cached full-graph returns through {last_available_date}."

    if use_live:
        live_state, live_message = try_live_full_graph_state()
        if live_state is not None:
            current_x = live_state
            data_source = "live_upstox_full_graph"
            data_message = live_message
        else:
            data_message = f"{live_message}; falling back to cached full-graph returns through {last_available_date}."

    idx = tickers.index(ticker)
    with torch.no_grad():
        final_probs = None
        for step in range(days):
            logits = model(current_x, edge_index, edge_weight)
            probs = torch.sigmoid(logits)
            final_probs = probs
            if step < days - 1:
                next_returns = (probs - 0.5) * 0.02
                current_x = torch.cat([current_x[:, 1:], next_returns.unsqueeze(-1)], dim=1)

    prob = float(final_probs[idx].item())
    tuned_precision = float(model_metrics["threshold_tuned"].get("precision", 0.0))
    confidence_decay = 0.85 ** (days - 1)
    confidence = tuned_precision * confidence_decay
    heuristic = days > 1
    sector = sector_lookup.get(ticker, {}).get("sector", "Unknown")
    live_evidence = build_live_evidence_for_ticker(ticker, sector)

    trained_model_output = {
        "ticker": ticker,
        "days": days,
        "probability_up": prob,
        "pUp": prob,
        "confidence": confidence,
        "confidence_label": "tuned-threshold precision" if not heuristic else "tuned-threshold precision with horizon decay",
        "heuristic": heuristic,
        "heuristic_label": None if not heuristic else "multi-day result uses naive iterative chaining; uncertainty compounds with each step",
        "tuned_threshold": tuned_threshold,
        "direction": "Likely Up" if prob >= tuned_threshold else "Likely Down",
        "data_source": data_source,
        "data_message": data_message,
        "last_available_date": last_available_date,
    }

    return {
        "ticker": ticker,
        "days": days,
        "trained_model_output": trained_model_output,
        "live_qualitative_evidence": live_evidence,
        "sector_membership": {
            "sector": sector,
            "source": sector_lookup.get(ticker, {}).get("source", "missing"),
            "cluster": cluster_payload(ticker),
        },
        "data_source": data_source,
        "data_message": data_message,
        "last_available_date": last_available_date,
        "cluster": cluster_payload(ticker),
        "graph": {
            "n_stocks": len(tickers),
            "edge_count": edge_count,
            "cluster_count": len(clusters),
            "cluster_modularity": cluster_modularity,
        },
        "metrics": model_metrics,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
