import sys
sys.path.insert(0, "/home/claude/cluster")

from pathlib import Path
import numpy as np
import pandas as pd

from causal_discovery import granger_edges
from graph_clustering import (
    weakly_connected_clusters, louvain_clusters, modularity_score, summarize_clusters
)

DATA_DIR = Path("/home/claude/cluster/data/nse_real")
STOCK_FILES = {
    "RELIANCE": "000_RELIANCE.csv", "TCS": "001_TCS.csv", "HDFCBANK": "003_HDFCBANK.csv",
    "INFY": "005_INFY.csv", "KOTAKBANK": "006_KOTAKBANK.csv", "ICICIBANK": "009_ICICIBANK.csv",
    "SBIN": "010_SBIN.csv", "HCLTECH": "015_HCLTECH.csv", "WIPRO": "017_WIPRO.csv",
    "AXISBANK": "018_AXISBANK.csv",
}
SECTOR = {
    "RELIANCE": "ENERGY", "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING", "SBIN": "BANKING",
}

def load_returns():
    series = {}
    for ticker, fname in STOCK_FILES.items():
        df = pd.read_csv(DATA_DIR / fname, parse_dates=["Date"]).sort_values("Date").set_index("Date")
        series[ticker] = df["Close"]
    prices = pd.DataFrame(series).dropna()
    return np.log(prices / prices.shift(1)).dropna()

if __name__ == "__main__":
    returns = load_returns()
    tickers = list(returns.columns)
    print(f"Loaded {len(returns)} real trading days, {len(tickers)} stocks\n")

    print("=== Step 1: Causal discovery (Granger, uncorrected alpha=0.05) ===")
    # uncorrected, as in the earlier real-data experiment, since Bonferroni
    # left too few edges (1) to demonstrate clustering meaningfully
    edges = granger_edges(returns, tickers, max_lag=2, alpha=0.05, bonferroni=False)
    print(f"Found {len(edges)} directed causal edges\n")

    print("=== Step 2a: Baseline -- weakly connected components ===")
    wc_clusters = weakly_connected_clusters(edges)
    summarize_clusters(wc_clusters, SECTOR)
    print()

    print("=== Step 2b: Real clustering -- Louvain community detection ===")
    lv_clusters = louvain_clusters(edges)
    summarize_clusters(lv_clusters, SECTOR)
    mod = modularity_score(edges, lv_clusters)
    print(f"\nModularity score: {mod:.3f}  (0 = no better than random grouping; "
          f"real-world networks with genuine community structure typically score 0.3+)")

    print("\n=== Step 3: Same analysis with PCMCI edges (more edges = more signal) ===")
    from pcmci_discovery import run_pcmci
    pcmci_edges = run_pcmci(returns, tau_max=2, pc_alpha=0.05)
    print(f"Found {len(pcmci_edges)} directed causal edges via PCMCI\n")
    pcmci_clusters = louvain_clusters(pcmci_edges)
    summarize_clusters(pcmci_clusters, SECTOR)
    pcmci_mod = modularity_score(pcmci_edges, pcmci_clusters)
    print(f"\nModularity score (PCMCI-based graph): {pcmci_mod:.3f}")
