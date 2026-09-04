"""
demo.py

Live-safe Causalyst demo.

Phase A trains or loads a cached causal graph from the full 2012-2021 history.
Phase B fetches the latest 5 trading days when Upstox support is available,
otherwise it runs a clearly labeled last-known-data demo from the cached CSVs.
The 5-day window is never used for causal discovery.
"""

from __future__ import annotations

import html
import importlib
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "causalyst"
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data" / "nse_real"
CACHE_PATH = ROOT / "causal_graph.pkl"
REPORT_PATH = ROOT / "demo_report.html"

for path in (ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


UNIVERSE = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "KOTAKBANK",
    "ICICIBANK",
    "SBIN",
    "HCLTECH",
    "WIPRO",
    "AXISBANK",
]

STOCK_FILES = {
    "RELIANCE": "000_RELIANCE.csv",
    "TCS": "001_TCS.csv",
    "HDFCBANK": "003_HDFCBANK.csv",
    "INFY": "005_INFY.csv",
    "KOTAKBANK": "006_KOTAKBANK.csv",
    "ICICIBANK": "009_ICICIBANK.csv",
    "SBIN": "010_SBIN.csv",
    "HCLTECH": "015_HCLTECH.csv",
    "WIPRO": "017_WIPRO.csv",
    "AXISBANK": "018_AXISBANK.csv",
}

SECTOR = {
    "RELIANCE": "ENERGY",
    "TCS": "IT",
    "INFY": "IT",
    "HCLTECH": "IT",
    "WIPRO": "IT",
    "HDFCBANK": "BANKING",
    "ICICIBANK": "BANKING",
    "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING",
    "SBIN": "BANKING",
}

TARGET_STOCK = "RELIANCE"
TRAINING_START = "2012-01-02"
TRAINING_END = "2021-12-31"
TRAINING_LABEL = f"{TRAINING_START} to {TRAINING_END}"
CACHE_VERSION = 2

CLUSTER_COLORS = ["#4BACC6", "#F79646", "#9BBB59", "#C0504D", "#8064A2"]
SECTOR_SHAPE = {"IT": "circle", "BANKING": "square", "ENERGY": "triangle"}


@dataclass
class DemoGraph:
    edges: Dict[Tuple[str, str], Dict[str, Any]]
    nodes: Sequence[str]

    def causal_parents_of(self, node: str) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            (src, info)
            for (src, tgt), info in self.edges.items()
            if tgt == node and info.get("edge_type", "causal") == "causal"
        ]

    def summary(self) -> str:
        return f"MarketGraph: {len(self.nodes)} companies, {len(self.edges)} causal edges"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def note(message: str) -> None:
    print(f"- {message}")


def load_cached_prices(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    series = {}
    for ticker in UNIVERSE:
        path = DATA_DIR / STOCK_FILES[ticker]
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
        close = df["Close"].astype(float)
        if start:
            close = close[close.index >= pd.Timestamp(start)]
        if end:
            close = close[close.index <= pd.Timestamp(end)]
        series[ticker] = close
    prices = pd.DataFrame(series).dropna()
    if prices.empty:
        raise RuntimeError(f"No cached CSV prices were found in {DATA_DIR}")
    return prices


def load_training_returns() -> pd.DataFrame:
    prices = load_cached_prices(TRAINING_START, TRAINING_END)
    return np.log(prices / prices.shift(1)).dropna()


def normalize_edges(raw_edges: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in raw_edges:
        src = item.get("from") or item.get("source")
        tgt = item.get("to") or item.get("target")
        if not src or not tgt:
            continue
        edges[(src, tgt)] = {
            "edge_type": "causal",
            "lag": int(item.get("lag", 1)),
            "p_value": float(item.get("p_value", 1.0)),
            "f_stat": item.get("f_stat"),
        }
    return edges


def discover_edges_from_full_history() -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], str]:
    try:
        pcmci = importlib.import_module("pcmci_discovery")
        returns = load_training_returns()
        raw = pcmci.run_pcmci(returns, tau_max=2, pc_alpha=0.05)
        edges = {
            (src, tgt): {
                "edge_type": "causal",
                "lag": int(info["lag"]),
                "p_value": float(info["p_value"]),
                "f_stat": info.get("f_stat"),
            }
            for (src, tgt), info in raw.items()
        }
        return edges, "live Phase A PCMCI run on full cached 2012-2021 history"
    except Exception as exc:
        reason = f"PCMCI training unavailable ({clean_error(exc)})"

    pcmci_json = PROJECT_DIR / "data" / "pcmci_results.json"
    if pcmci_json.exists():
        with pcmci_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return normalize_edges(payload.get("full_period_edges", [])), (
            f"cached Phase A PCMCI result from {pcmci_json.name}; {reason}"
        )

    real_json = PROJECT_DIR / "data" / "real_data_results.json"
    if real_json.exists():
        with real_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        raw_edges = payload.get("full_period", {}).get("granger_uncorrected_edges", [])
        return normalize_edges(raw_edges), (
            f"cached Phase A Granger result from {real_json.name}; {reason}"
        )

    raise RuntimeError(
        "No Phase A graph is available. Expected pcmci_results.json, "
        "real_data_results.json, or installable PCMCI dependencies."
    )


def build_project_graph(edges: Dict[Tuple[str, str], Dict[str, Any]]) -> Tuple[Any, str]:
    node_types = {ticker: "COMPANY" for ticker in UNIVERSE}
    try:
        graph_builder = importlib.import_module("graph_builder")
        mg = graph_builder.build_graph_from_discovery(edges, node_types)
        return mg, "graph_builder.MarketGraph.causal_parents_of()"
    except Exception as exc:
        return DemoGraph(edges, UNIVERSE), (
            "demo-compatible causal_parents_of() wrapper "
            f"(graph_builder unavailable: {clean_error(exc)})"
        )


def edge_weight(info: Dict[str, Any]) -> float:
    return -math.log10(max(float(info.get("p_value", 1.0)), 1e-12))


def compute_clusters(edges: Dict[Tuple[str, str], Dict[str, Any]]) -> Tuple[List[set], str]:
    try:
        graph_clustering = importlib.import_module("graph_clustering")
        clusters = [set(c) for c in graph_clustering.louvain_clusters(edges)]
        seen = set().union(*clusters) if clusters else set()
        clusters.extend({ticker} for ticker in UNIVERSE if ticker not in seen)
        return sorted(clusters, key=lambda c: (-len(c), sorted(c))), "Louvain via graph_clustering.py"
    except Exception as exc:
        graph_reason = clean_error(exc)

    try:
        from sklearn.cluster import SpectralClustering

        nodes = UNIVERSE[:]
        idx = {node: i for i, node in enumerate(nodes)}
        affinity = np.zeros((len(nodes), len(nodes)))
        for (src, tgt), info in edges.items():
            if src not in idx or tgt not in idx:
                continue
            i, j = idx[src], idx[tgt]
            w = edge_weight(info)
            affinity[i, j] += w
            affinity[j, i] += w
        if np.count_nonzero(affinity) == 0:
            return [{ticker} for ticker in nodes], (
                f"singleton fallback; graph_clustering unavailable: {graph_reason}"
            )
        labels = SpectralClustering(
            n_clusters=3,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        ).fit_predict(affinity)
        buckets: Dict[int, set] = {}
        for node, label in zip(nodes, labels):
            buckets.setdefault(int(label), set()).add(node)
        return sorted(buckets.values(), key=lambda c: (-len(c), sorted(c))), (
            "spectral clustering fallback; graph_clustering unavailable: "
            f"{graph_reason}"
        )
    except Exception as exc:
        clusters = weak_components(edges)
        return clusters, (
            "weakly connected fallback; graph_clustering and sklearn unavailable: "
            f"{graph_reason}; {clean_error(exc)}"
        )


def weak_components(edges: Dict[Tuple[str, str], Dict[str, Any]]) -> List[set]:
    adj = {ticker: set() for ticker in UNIVERSE}
    for src, tgt in edges:
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set()).add(src)
    seen = set()
    clusters = []
    for node in UNIVERSE:
        if node in seen:
            continue
        stack = [node]
        group = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            group.add(cur)
            stack.extend(sorted(adj.get(cur, set()) - seen))
        clusters.append(group)
    return sorted(clusters, key=lambda c: (-len(c), sorted(c)))


def save_graph_cache(
    edges: Dict[Tuple[str, str], Dict[str, Any]],
    clusters: List[set],
    edge_source: str,
    cluster_method: str,
) -> None:
    payload = {
        "version": CACHE_VERSION,
        "target_stock": TARGET_STOCK,
        "training_window": TRAINING_LABEL,
        "edge_source": edge_source,
        "cluster_method": cluster_method,
        "edges": [
            {"from": src, "to": tgt, **info}
            for (src, tgt), info in sorted(edges.items())
        ],
        "clusters": [sorted(c) for c in clusters],
    }
    with CACHE_PATH.open("wb") as f:
        pickle.dump(payload, f)


def load_or_train_phase_a() -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[set], Dict[str, str]]:
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("version") == CACHE_VERSION:
                edges = normalize_edges(payload.get("edges", []))
                clusters = [set(c) for c in payload.get("clusters", [])]
                cluster_method = payload.get("cluster_method", "cached clusters")
                cache_status = "USING CACHE"
                if not cluster_method.startswith("Louvain"):
                    refreshed_clusters, refreshed_method = compute_clusters(edges)
                    if refreshed_method.startswith("Louvain"):
                        clusters = refreshed_clusters
                        cluster_method = refreshed_method
                        save_graph_cache(
                            edges,
                            clusters,
                            payload.get("edge_source", "cached Phase A graph"),
                            cluster_method,
                        )
                        cache_status = "REFRESHED CACHE with Louvain clusters"
                meta = {
                    "edge_source": (
                        f"cached graph from {CACHE_PATH.name}; original source: "
                        f"{payload.get('edge_source', 'unknown')}"
                    ),
                    "cluster_method": cluster_method,
                    "cache_status": cache_status,
                }
                return edges, clusters, meta
        except Exception as exc:
            note(f"Ignoring unreadable graph cache: {clean_error(exc)}")

    edges, edge_source = discover_edges_from_full_history()
    clusters, cluster_method = compute_clusters(edges)
    save_graph_cache(edges, clusters, edge_source, cluster_method)
    meta = {
        "edge_source": edge_source,
        "cluster_method": cluster_method,
        "cache_status": f"CREATED CACHE at {CACHE_PATH.name}",
    }
    return edges, clusters, meta


def try_upstox_live_prices(days: int = 5) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        upstox_fetch = importlib.import_module("upstox_fetch")
    except Exception as exc:
        return None, f"Upstox live fetch unavailable: upstox_fetch.py not found or not importable ({clean_error(exc)})"

    candidate_calls = [
        ("fetch_recent_prices", (UNIVERSE, days), {}),
        ("fetch_last_n_trading_days", (UNIVERSE, days), {}),
        ("fetch_recent_ohlc", (UNIVERSE, days), {}),
        ("fetch_universe_ohlc", (UNIVERSE,), {"days": days}),
        ("fetch_universe_history", (UNIVERSE,), {"days": days}),
    ]
    for func_name, args, kwargs in candidate_calls:
        func = getattr(upstox_fetch, func_name, None)
        if not callable(func):
            continue
        try:
            raw = func(*args, **kwargs)
            prices = normalize_live_price_payload(raw)
            if prices is not None and len(prices) >= 2:
                return prices.tail(days), f"LIVE DATA via upstox_fetch.{func_name}()"
        except Exception as exc:
            return None, f"Upstox call failed in {func_name}(): {clean_error(exc)}"

    return None, (
        "Upstox live fetch unavailable: upstox_fetch.py has no supported "
        "recent-price function name for this demo."
    )


def normalize_live_price_payload(raw: Any) -> Optional[pd.DataFrame]:
    if raw is None:
        return None
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        if all(ticker in df.columns for ticker in UNIVERSE):
            return df[UNIVERSE].astype(float).dropna()
        if {"ticker", "date", "close"}.issubset({c.lower() for c in df.columns}):
            lower = {c.lower(): c for c in df.columns}
            tidy = df.rename(
                columns={
                    lower["ticker"]: "ticker",
                    lower["date"]: "date",
                    lower["close"]: "close",
                }
            )
            tidy["date"] = pd.to_datetime(tidy["date"])
            return tidy.pivot(index="date", columns="ticker", values="close")[UNIVERSE].dropna()
    if isinstance(raw, dict):
        frames = {}
        for ticker, value in raw.items():
            if ticker not in UNIVERSE:
                continue
            if isinstance(value, pd.DataFrame):
                frame = value.copy()
                date_col = "Date" if "Date" in frame.columns else "date" if "date" in frame.columns else None
                close_col = "Close" if "Close" in frame.columns else "close" if "close" in frame.columns else None
                if date_col and close_col:
                    frame[date_col] = pd.to_datetime(frame[date_col])
                    frames[ticker] = frame.sort_values(date_col).set_index(date_col)[close_col].astype(float)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                frame = pd.DataFrame(value)
                if {"date", "close"}.issubset(frame.columns):
                    frame["date"] = pd.to_datetime(frame["date"])
                    frames[ticker] = frame.sort_values("date").set_index("date")["close"].astype(float)
        if frames:
            return pd.DataFrame(frames).dropna()
    return None


def load_phase_b_prices() -> Tuple[pd.DataFrame, str, str]:
    live_prices, live_message = try_upstox_live_prices(days=5)
    if live_prices is not None:
        return live_prices, "LIVE DATA", live_message
    cached = load_cached_prices(TRAINING_START, TRAINING_END).tail(5)
    return cached, "FALLBACK DATA", (
        f"{live_message}. Using last known cached CSV trading days from "
        f"{cached.index.min().date()} to {cached.index.max().date()}."
    )


def cluster_for_stock(clusters: List[set], ticker: str) -> Tuple[int, set]:
    for i, cluster in enumerate(clusters, start=1):
        if ticker in cluster:
            return i, cluster
    return 0, {ticker}


def format_parent_evidence(parents: Sequence[Tuple[str, Dict[str, Any]]], target: str) -> List[str]:
    if not parents:
        return [f"No Phase A causal parents were found for {target} in the trained graph."]
    evidence = []
    for src, info in sorted(parents, key=lambda item: item[1].get("p_value", 1.0)):
        evidence.append(
            f"{src} return causally precedes {target} return at lag={info.get('lag', '?')} "
            f"trading day(s), p={float(info.get('p_value', 1.0)):.5f}."
        )
    return evidence


def build_agent_evidence(
    target: str,
    prices: pd.DataFrame,
    parent_evidence: List[str],
    data_label: str,
) -> Dict[str, List[str]]:
    target_prices = prices[target]
    move_pct = pct_move(target_prices.iloc[0], target_prices.iloc[-1])
    dates = f"{prices.index.min().date()} to {prices.index.max().date()}"
    price_fact = (
        f"{data_label}: {target} close moved from {target_prices.iloc[0]:.2f} "
        f"to {target_prices.iloc[-1]:.2f} over {dates} ({move_pct:+.2f}%)."
    )
    no_direct_flow = (
        "No FII/DII node is available in this CSV-based real-data graph; "
        "do not describe this as direct institutional-flow evidence."
    )
    return {
        "fundamentals": [
            price_fact,
            "No live earnings, guidance, rating, or corporate-action feed is integrated in this demo.",
        ],
        "sentiment": [
            "No live news or social-sentiment feed is integrated in this demo.",
            "Use the causal graph evidence only if it supports a market-spillover interpretation.",
        ],
        "macro": parent_evidence + [
            "These are lagged cross-stock return links from Phase A, not a 5-day causal rediscovery."
        ],
        "institutional_flow": [no_direct_flow] + parent_evidence,
    }


def run_debate_safely(evidence: Dict[str, List[str]]) -> Tuple[Dict[str, str], str]:
    try:
        agents = importlib.import_module("agents")
        transcript = agents.run_debate(evidence)
        return transcript, "LIVE CLAUDE API"
    except Exception as exc:
        return cached_debate_transcript(evidence, clean_error(exc)), (
            "FALLBACK TRANSCRIPT: debate layer unavailable; "
            f"showing cached example transcript ({clean_error(exc)})"
        )


def cached_debate_transcript(evidence: Dict[str, List[str]], reason: str) -> Dict[str, str]:
    parent_line = evidence.get("macro", ["No parent evidence."])[0]
    return {
        "fundamentals": json.dumps(
            {
                "stance": "neutral",
                "confidence": 0.42,
                "reasoning": "The 5-day price move is visible, but no earnings, guidance, or balance-sheet evidence is available in the demo feed.",
            }
        ),
        "sentiment": json.dumps(
            {
                "stance": "neutral",
                "confidence": 0.35,
                "reasoning": "No news or social sentiment signal is integrated, so sentiment cannot responsibly explain the move.",
            }
        ),
        "macro": json.dumps(
            {
                "stance": "cautiously directional",
                "confidence": 0.56,
                "reasoning": parent_line[:180],
            }
        ),
        "institutional_flow": json.dumps(
            {
                "stance": "neutral",
                "confidence": 0.32,
                "reasoning": "The graph has no FII/DII node, so any flow claim would be a proxy interpretation rather than direct evidence.",
            }
        ),
        "contrarian": json.dumps(
            {
                "stance": "contrarian_challenge",
                "confidence": 0.61,
                "reasoning": "The causal evidence is historical and price-only; it should not be overread as a live causal explanation for five days.",
            }
        ),
        "synthesis": json.dumps(
            {
                "final_stance": "neutral",
                "confidence": 0.46,
                "plain_english_explanation": (
                    "The demo shows a valid two-phase pipeline: historical causal graph first, "
                    "then recent price evidence and agent reasoning. The final call remains neutral "
                    "because the live debate API was unavailable and the real-data graph contains "
                    "price-return links, not direct institutional-flow or news signals."
                ),
                "key_disagreement": f"Claude debate unavailable: {reason[:80]}",
            }
        ),
    }


def pct_move(start: float, end: float) -> float:
    return ((end / start) - 1.0) * 100.0


def clean_error(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", " ")
    return text or exc.__class__.__name__


def parse_jsonish(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def extract_final_confidence(transcript: Dict[str, str]) -> str:
    synthesis = parse_jsonish(transcript.get("synthesis", "{}"))
    confidence = synthesis.get("confidence")
    if isinstance(confidence, (int, float)):
        return f"{confidence:.2f}"
    return "n/a"


def make_price_chart_svg(prices: pd.Series, title: str) -> str:
    width, height = 760, 300
    left, right, top, bottom = 70, 24, 40, 48
    vals = prices.astype(float).to_numpy()
    dates = [d.strftime("%b %d") for d in prices.index]
    lo, hi = float(vals.min()), float(vals.max())
    pad = max((hi - lo) * 0.12, hi * 0.002, 1.0)
    lo -= pad
    hi += pad

    def x(i: int) -> float:
        return left + i * (width - left - right) / max(len(vals) - 1, 1)

    def y(v: float) -> float:
        return top + (hi - v) * (height - top - bottom) / max(hi - lo, 1e-9)

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
    circles = []
    labels = []
    for i, v in enumerate(vals):
        circles.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="5" fill="#002060"/>')
        labels.append(
            f'<text x="{x(i):.1f}" y="{height - 18}" text-anchor="middle" '
            f'font-size="12" fill="#333">{html.escape(dates[i])}</text>'
        )
    y_ticks = []
    for frac in (0, 0.5, 1):
        value = lo + (hi - lo) * frac
        yy = y(value)
        y_ticks.append(
            f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" '
            f'stroke="#e6e8eb"/>'
            f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-size="12" '
            f'fill="#555">{value:.0f}</text>'
        )
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <text x="{left}" y="24" font-size="17" font-weight="700" fill="#002060">{html.escape(title)}</text>
  {''.join(y_ticks)}
  <polyline points="{points}" fill="none" stroke="#4BACC6" stroke-width="4" stroke-linejoin="round"/>
  {''.join(circles)}
  {''.join(labels)}
</svg>
"""


def make_cluster_graph_svg(
    edges: Dict[Tuple[str, str], Dict[str, Any]],
    clusters: List[set],
    target: str,
) -> str:
    width, height = 860, 520
    positions = {
        "RELIANCE": (430, 80),
        "TCS": (160, 170),
        "INFY": (250, 320),
        "HCLTECH": (120, 410),
        "WIPRO": (330, 430),
        "HDFCBANK": (610, 165),
        "ICICIBANK": (710, 285),
        "KOTAKBANK": (560, 360),
        "AXISBANK": (745, 430),
        "SBIN": (470, 455),
    }
    cluster_index = {}
    for i, cluster in enumerate(clusters):
        for node in cluster:
            cluster_index[node] = i

    weights = [edge_weight(info) for info in edges.values()]
    max_w = max(weights) if weights else 1.0
    edge_lines = []
    for (src, tgt), info in sorted(edges.items(), key=lambda item: edge_weight(item[1])):
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        stroke = 0.7 + 3.2 * edge_weight(info) / max_w
        opacity = 0.22 + 0.42 * edge_weight(info) / max_w
        edge_lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#888" stroke-width="{stroke:.2f}" opacity="{opacity:.2f}">'
            f'<title>{html.escape(src)} -> {html.escape(tgt)}, lag={info.get("lag")}, '
            f'p={float(info.get("p_value", 1.0)):.5f}</title></line>'
        )

    node_shapes = []
    for node, (x, y) in positions.items():
        color = CLUSTER_COLORS[cluster_index.get(node, 0) % len(CLUSTER_COLORS)]
        sector = SECTOR.get(node, "?")
        shape = SECTOR_SHAPE.get(sector, "circle")
        border = "#111" if node == target else "#333"
        stroke_width = "4" if node == target else "1.6"
        if shape == "square":
            node_shapes.append(
                f'<rect x="{x-28}" y="{y-28}" width="56" height="56" rx="3" '
                f'fill="{color}" stroke="{border}" stroke-width="{stroke_width}"/>'
            )
        elif shape == "triangle":
            pts = f"{x},{y-33} {x-33},{y+28} {x+33},{y+28}"
            node_shapes.append(
                f'<polygon points="{pts}" fill="{color}" stroke="{border}" '
                f'stroke-width="{stroke_width}"/>'
            )
        else:
            node_shapes.append(
                f'<circle cx="{x}" cy="{y}" r="31" fill="{color}" stroke="{border}" '
                f'stroke-width="{stroke_width}"/>'
            )
        node_shapes.append(
            f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="12" '
            f'font-weight="700" fill="#111">{html.escape(node)}</text>'
        )

    legend = """
  <g transform="translate(28,28)">
    <circle cx="0" cy="0" r="9" fill="#aaa" stroke="#333"/><text x="18" y="4" font-size="12">IT</text>
    <rect x="72" y="-9" width="18" height="18" rx="2" fill="#aaa" stroke="#333"/><text x="98" y="4" font-size="12">Banking</text>
    <polygon points="185,-11 174,10 196,10" fill="#aaa" stroke="#333"/><text x="204" y="4" font-size="12">Energy</text>
  </g>
"""
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Causal graph and clusters">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <text x="28" y="500" font-size="12" fill="#555">Node color = discovered cluster. Node shape = sector. Edge width = -log10(p-value). {html.escape(target)} is outlined.</text>
  {legend}
  {''.join(edge_lines)}
  {''.join(node_shapes)}
</svg>
"""


def render_transcript_html(transcript: Dict[str, str]) -> str:
    parts = []
    for role in ["fundamentals", "sentiment", "macro", "institutional_flow", "contrarian", "synthesis"]:
        parsed = parse_jsonish(transcript.get(role, ""))
        if "raw" in parsed:
            body = f"<pre>{html.escape(str(parsed['raw']))}</pre>"
        else:
            rows = "".join(
                f"<p><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</p>"
                for k, v in parsed.items()
            )
            body = rows
        parts.append(f"<section class=\"panel\"><h3>{html.escape(role.replace('_', ' ').title())}</h3>{body}</section>")
    return "\n".join(parts)


def write_html_report(
    prices: pd.DataFrame,
    edges: Dict[Tuple[str, str], Dict[str, Any]],
    clusters: List[set],
    target: str,
    parent_evidence: List[str],
    transcript: Dict[str, str],
    phase_a_meta: Dict[str, str],
    data_label: str,
    data_message: str,
    debate_label: str,
) -> None:
    target_prices = prices[target]
    price_svg = make_price_chart_svg(target_prices, f"{target} 5-Day Close")
    graph_svg = make_cluster_graph_svg(edges, clusters, target)
    cluster_no, cluster = cluster_for_stock(clusters, target)
    evidence_html = "".join(f"<li>{html.escape(e)}</li>" for e in parent_evidence)
    final_confidence = extract_final_confidence(transcript)
    move = pct_move(target_prices.iloc[0], target_prices.iloc[-1])
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Causalyst Demo Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f6f8fb; }}
    header {{ background: #002060; color: white; padding: 26px 34px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ color: #002060; margin: 0 0 12px; font-size: 21px; }}
    h3 {{ margin: 0 0 10px; color: #002060; font-size: 16px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 26px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 18px; }}
    .panel {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
    .badge {{ display: inline-block; padding: 5px 9px; border-radius: 5px; font-size: 12px; font-weight: 700; background: #e8f2ff; color: #002060; margin-right: 8px; }}
    .warn {{ background: #fff4df; color: #7a4b00; }}
    .ok {{ background: #e7f6ef; color: #14532d; }}
    ul {{ margin-top: 8px; }}
    pre {{ white-space: pre-wrap; font-family: Consolas, monospace; font-size: 13px; }}
    svg {{ width: 100%; height: auto; }}
  </style>
</head>
<body>
  <header>
    <h1>Causalyst Live Demo Report</h1>
    <div>Target: {html.escape(target)} | Final confidence: {html.escape(final_confidence)} | 5-day move: {move:+.2f}%</div>
  </header>
  <main>
    <section class="panel">
      <h2>Demo Status</h2>
      <p><span class="badge ok">Phase A</span>{html.escape(phase_a_meta["cache_status"])} from {html.escape(TRAINING_LABEL)}.</p>
      <p><span class="badge">Graph</span>{html.escape(phase_a_meta["edge_source"])}</p>
      <p><span class="badge">Clusters</span>{html.escape(phase_a_meta["cluster_method"])}</p>
      <p><span class="badge {'warn' if data_label != 'LIVE DATA' else 'ok'}">{html.escape(data_label)}</span>{html.escape(data_message)}</p>
      <p><span class="badge {'warn' if debate_label.startswith('FALLBACK') else 'ok'}">Debate</span>{html.escape(debate_label)}</p>
    </section>

    <div class="grid">
      <section class="panel">
        <h2>5-Day Price Move</h2>
        {price_svg}
      </section>
      <section class="panel">
        <h2>Causal Evidence Used</h2>
        <p>{html.escape(target)} belongs to cluster {cluster_no}: {html.escape(', '.join(sorted(cluster)))}</p>
        <ul>{evidence_html}</ul>
      </section>
    </div>

    <section class="panel">
      <h2>Causal Graph And Clusters</h2>
      {graph_svg}
    </section>

    <section>
      <h2>Debate Transcript</h2>
      {render_transcript_html(transcript)}
    </section>
  </main>
</body>
</html>
"""
    REPORT_PATH.write_text(html_doc, encoding="utf-8")


def main() -> None:
    section("Causalyst Demo: Causal Multi-Agent Stock Prediction")
    note(f"Universe: {', '.join(UNIVERSE)}")
    note(f"Featured stock: {TARGET_STOCK}")
    note("Guardrail: the 5-day live/demo window is NOT used for causal discovery.")

    section("PHASE A - Training Graph (Full Historical Window)")
    edges, clusters, phase_a_meta = load_or_train_phase_a()
    graph, graph_method = build_project_graph(edges)
    note(f"Training window: {TRAINING_LABEL}")
    note(f"Graph source: {phase_a_meta['edge_source']}")
    note(f"Cache status: {phase_a_meta['cache_status']}")
    note(f"Graph interface: {graph_method}")
    note(f"Cluster method: {phase_a_meta['cluster_method']}")
    note(graph.summary() if hasattr(graph, "summary") else f"{len(edges)} causal edges")

    section("PHASE B - 5-Day Live Demo Window")
    prices, data_label, data_message = load_phase_b_prices()
    target_prices = prices[TARGET_STOCK]
    move = pct_move(target_prices.iloc[0], target_prices.iloc[-1])
    cluster_no, cluster = cluster_for_stock(clusters, TARGET_STOCK)
    parents = graph.causal_parents_of(TARGET_STOCK)
    parent_evidence = format_parent_evidence(parents, TARGET_STOCK)

    note(f"Data status: {data_label}")
    note(data_message)
    note(
        f"{TARGET_STOCK} close: {target_prices.iloc[0]:.2f} on {prices.index.min().date()} "
        f"-> {target_prices.iloc[-1]:.2f} on {prices.index.max().date()} ({move:+.2f}%)"
    )
    note(f"Cluster {cluster_no}: {', '.join(sorted(cluster))}")
    print("\nCausal parents used as evidence:")
    for item in parent_evidence:
        print(f"  - {item}")

    section("Multi-Agent Debate")
    evidence = build_agent_evidence(TARGET_STOCK, prices, parent_evidence, data_label)
    transcript, debate_label = run_debate_safely(evidence)
    note(f"Debate status: {debate_label}")
    for role in ["fundamentals", "sentiment", "macro", "institutional_flow", "contrarian", "synthesis"]:
        parsed = parse_jsonish(transcript.get(role, ""))
        print(f"\n[{role.replace('_', ' ').title()}]")
        if "raw" in parsed:
            print(parsed["raw"])
        else:
            for key, value in parsed.items():
                print(f"{key}: {value}")
    note(f"Final confidence score: {extract_final_confidence(transcript)}")

    section("HTML Report")
    write_html_report(
        prices=prices,
        edges=edges,
        clusters=clusters,
        target=TARGET_STOCK,
        parent_evidence=parent_evidence,
        transcript=transcript,
        phase_a_meta=phase_a_meta,
        data_label=data_label,
        data_message=data_message,
        debate_label=debate_label,
    )
    note(f"Report written to: {REPORT_PATH}")
    note("Open demo_report.html before or during the presentation for a screen-friendly view.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as exc:
        print("\nA recoverable demo setup problem occurred.")
        print(f"Reason: {clean_error(exc)}")
        print("No raw traceback shown; fix the message above and rerun python demo.py.")
