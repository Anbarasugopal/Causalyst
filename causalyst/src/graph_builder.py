"""
graph_builder.py

Stage 2 of the pipeline: build the dynamic market graph.

Key design decision (the paper's core contribution): FII and DII flow are
modeled as NODES in the graph -- distinct actors with their own time series --
not merely as edge-attributes connecting companies that share institutional
ownership. This lets Stage 3 (causal discovery) test hypotheses like
"does FII_flow Granger-cause RELIANCE_return" directly, the same way it tests
"does TCS_return Granger-cause INFY_return".

Node types:
    - COMPANY   (e.g., RELIANCE, TCS)
    - SECTOR    (e.g., IT, ENERGY)     -- static membership edges
    - FLOW      (e.g., FII, DII)       -- the institutional-flow contribution
    - MACRO     (e.g., REPO_RATE, CRUDE_OIL)

Edge types:
    - CAUSAL (directed, from Stage 3, carries lag + p-value)
    - SECTOR_MEMBERSHIP (static, undirected)
    - SHARED_OWNERSHIP (dynamic, undirected, from shareholding-pattern filings)

This module is intentionally lightweight (networkx) rather than requiring
PyTorch Geometric, so the graph structure and causal-discovery validation can
be built, tested, and visualized without a GPU. Swap the `to_pyg_data()`
exporter at the bottom for real GNN training once you have PyTorch Geometric
installed and real multi-entity data.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import pandas as pd
import requests

OFFICIAL_NSE_INDEX_LISTS = [
    "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_niftymidcap50list.csv",
    "https://archives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
    "https://archives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
    "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
]

SECTOR_CACHE_PATH = Path(__file__).resolve().parents[2] / "sector_lookup.json"


def _normalize_ticker(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace(" ", "")


def _fetch_nse_index_sector_table() -> pd.DataFrame:
    frames = []
    for url in OFFICIAL_NSE_INDEX_LISTS:
        try:
            text = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
            df = pd.read_csv(pd.io.common.StringIO(text))
            if {"Symbol", "Industry"}.issubset(df.columns):
                df = df[["Symbol", "Industry"]].copy()
                df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
                df["Industry"] = df["Industry"].astype(str).str.strip()
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["Symbol", "Industry", "source"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Symbol"]).copy()
    combined["source"] = "official_nse_index_csv"
    return combined


def build_sector_lookup_for_tickers(tickers: Iterable[str], cache_path: Path = SECTOR_CACHE_PATH) -> Dict[str, Dict[str, str]]:
    """Build a ticker->sector mapping from official NSE index constituent CSVs.

    The historical 376-stock graph is not an exact match to the current published
    NSE constituent lists, so the remaining symbols are intentionally left as
    "Unknown" rather than guessed from ticker names or other approximate sources.
    """
    tickers = [_normalize_ticker(t) for t in tickers]
    if cache_path.exists():
        try:
            mapping = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(mapping, dict) and all(isinstance(v, dict) for v in mapping.values()):
                return {k.upper(): v for k, v in mapping.items() if k}
        except Exception:
            pass

    official = _fetch_nse_index_sector_table()
    lookup = {}
    for _, row in official.iterrows():
        sym = _normalize_ticker(row["Symbol"])
        lookup[sym] = {
            "sector": row.get("Industry", "Unknown"),
            "source": "official_nse_index_csv",
        }

    for symbol in tickers:
        lookup.setdefault(symbol, {"sector": "Unknown", "source": "missing"})

    cache_path.write_text(json.dumps({k: v for k, v in sorted(lookup.items())}, indent=2), encoding="utf-8")
    return lookup


def add_sector_membership_to_graph(graph: "MarketGraph", tickers: Iterable[str], sector_lookup: Dict[str, Dict[str, str]]):
    for ticker in tickers:
        ticker_upper = _normalize_ticker(ticker)
        sector = sector_lookup.get(ticker_upper, {}).get("sector", "Unknown")
        graph.add_sector_membership(ticker_upper, sector)


class MarketGraph:
    def __init__(self):
        self.g = nx.DiGraph()

    def add_node(self, name: str, node_type: str, **attrs):
        self.g.add_node(name, node_type=node_type, **attrs)

    def add_causal_edge(self, source: str, target: str, lag: int, p_value: float, f_stat: float = None):
        self.g.add_edge(
            source, target,
            edge_type="causal",
            lag=lag,
            p_value=p_value,
            f_stat=f_stat,
        )

    def add_sector_membership(self, company: str, sector: str):
        company = _normalize_ticker(company)
        sector = str(sector or "Unknown").strip() or "Unknown"
        self.g.add_edge(company, sector, edge_type="sector_membership")
        self.g.add_edge(sector, company, edge_type="sector_membership")

    def add_shared_ownership(self, company_a: str, company_b: str, weight: float):
        self.g.add_edge(company_a, company_b, edge_type="shared_ownership", weight=weight)
        self.g.add_edge(company_b, company_a, edge_type="shared_ownership", weight=weight)

    def causal_parents_of(self, node: str) -> List[Tuple[str, Dict]]:
        """Everything that Stage 3 found to causally precede `node` -- this is
        exactly what Stage 4 agents are handed as their evidence set."""
        return [
            (src, self.g.edges[src, node])
            for src in self.g.predecessors(node)
            if self.g.edges[src, node].get("edge_type") == "causal"
        ]

    def summary(self) -> str:
        n_flow = sum(1 for _, d in self.g.nodes(data=True) if d.get("node_type") == "FLOW")
        n_company = sum(1 for _, d in self.g.nodes(data=True) if d.get("node_type") == "COMPANY")
        n_causal_edges = sum(1 for _, _, d in self.g.edges(data=True) if d.get("edge_type") == "causal")
        return (
            f"MarketGraph: {self.g.number_of_nodes()} nodes "
            f"({n_company} companies, {n_flow} institutional-flow nodes), "
            f"{n_causal_edges} causal edges"
        )


def build_graph_from_discovery(
    causal_edges: Dict[Tuple[str, str], Dict],
    node_types: Dict[str, str],
) -> MarketGraph:
    """Assemble a MarketGraph from the output of causal_discovery.granger_edges()."""
    mg = MarketGraph()
    for node, ntype in node_types.items():
        mg.add_node(node, ntype)
    for (src, tgt), info in causal_edges.items():
        mg.add_causal_edge(src, tgt, lag=info["lag"], p_value=info["p_value"], f_stat=info.get("f_stat"))
    return mg


if __name__ == "__main__":
    from synthetic_data import generate_synthetic_market
    from causal_discovery import granger_edges

    df = generate_synthetic_market()
    cols = ["macro_rate", "fii_flow", "dii_flow", "sentiment_A", "stock_A_return", "stock_B_return"]
    edges = granger_edges(df, cols, max_lag=3)

    node_types = {
        "macro_rate": "MACRO",
        "fii_flow": "FLOW",
        "dii_flow": "FLOW",
        "sentiment_A": "MACRO",
        "stock_A_return": "COMPANY",
        "stock_B_return": "COMPANY",
    }
    mg = build_graph_from_discovery(edges, node_types)
    print(mg.summary())
    print("\nCausal parents of stock_A_return (this is what gets handed to the agents):")
    for src, edge in mg.causal_parents_of("stock_A_return"):
        print(f"  {src} -> stock_A_return  (lag={edge['lag']}, p={edge['p_value']:.4f})")
