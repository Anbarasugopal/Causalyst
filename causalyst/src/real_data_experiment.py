"""
real_data_experiment.py

PRELIMINARY REAL-DATA CASE STUDY.

Uses REAL daily OHLCV data for 10 actual NSE-listed stocks (5 IT-sector,
5 Banking-sector), Jan 2012 - Dec 2021, sourced from a public GitHub mirror
of NSE data (Ratnesh-bhosale/NIFTY500_dataset; original data attributed to
NSE India). This is genuine market data -- not synthetic -- but note what
this experiment does NOT cover (be honest about this in the paper):

    - No FII/DII institutional-flow data (could not be freely obtained in
      bulk historical form -- see scripts/fii_dii_scraper_TEMPLATE.py for
      a forward-collecting scraper you would need to run going forward)
    - No prediction / backtest -- this only tests the CAUSAL DISCOVERY step
      (Stage 3), not the full pipeline
    - Only 10 stocks across 2 sectors -- not a full Nifty 50/500 universe

What this DOES legitimately test: does Granger causality (our method) find
a sparser, more sector-structured graph than naive correlation on REAL
stock returns, and is the discovered structure STABLE across two different
5-year sub-periods (a real robustness check, not assumed)?

Stocks:
    IT:      TCS, INFY, HCLTECH, WIPRO   (+ RELIANCE as a cross-sector control)
    Banking: HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, SBIN
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_discovery import correlation_edges, granger_edges

DATA_DIR = Path("/home/claude/causalyst/data/nse_real")

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
    "RELIANCE": "ENERGY/CONGLOMERATE",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING", "SBIN": "BANKING",
}


def load_returns() -> pd.DataFrame:
    """Load real close prices for all 10 stocks, align on common trading
    days, and compute daily log returns."""
    series = {}
    for ticker, fname in STOCK_FILES.items():
        df = pd.read_csv(DATA_DIR / fname, parse_dates=["Date"])
        df = df.sort_values("Date").set_index("Date")
        series[ticker] = df["Close"]
    prices = pd.DataFrame(series).dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def edge_sector_breakdown(edges, sector_map):
    within, cross = 0, 0
    for (a, b) in edges:
        if sector_map[a] == sector_map[b]:
            within += 1
        else:
            cross += 1
    return {"within_sector": within, "cross_sector": cross}


def jaccard(set_a, set_b) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def run():
    returns = load_returns()
    tickers = list(returns.columns)
    n_days = len(returns)
    print(f"Loaded {n_days} trading days of REAL returns for {len(tickers)} stocks: {tickers}")
    print(f"Date range: {returns.index.min().date()} to {returns.index.max().date()}\n")

    results = {"n_trading_days": n_days, "tickers": tickers}

    # --- Full-period comparison: correlation vs Granger ---
    print("=== Full period: correlation-only baseline ===")
    corr_full = correlation_edges(returns, tickers, threshold=0.3)
    print(f"Edges found: {len(corr_full)}  |  sector breakdown: {edge_sector_breakdown(corr_full, SECTOR)}")

    print("\n=== Full period: Granger causality (our method, Bonferroni-corrected) ===")
    granger_full = granger_edges(returns, tickers, max_lag=2, alpha=0.05, bonferroni=True)
    print(f"Edges found: {len(granger_full)}  |  sector breakdown: {edge_sector_breakdown(granger_full, SECTOR)}")
    for (a, b), info in sorted(granger_full.items(), key=lambda x: x[1]["p_value"]):
        print(f"  {a} -> {b}  (lag={info['lag']}, p={info['p_value']:.5f})")

    results["full_period"] = {
        "correlation_edge_count": len(corr_full),
        "correlation_sector_breakdown": edge_sector_breakdown(corr_full, SECTOR),
        "granger_edge_count": len(granger_full),
        "granger_sector_breakdown": edge_sector_breakdown(granger_full, SECTOR),
        "granger_edges": [
            {"from": a, "to": b, "lag": info["lag"], "p_value": round(info["p_value"], 5)}
            for (a, b), info in granger_full.items()
        ],
    }

    # --- Stability check: split into two 5-year halves, compare discovered edges ---
    midpoint = returns.index[n_days // 2]
    first_half = returns[returns.index < midpoint]
    second_half = returns[returns.index >= midpoint]
    print(f"\n=== Stability check: {first_half.index.min().date()}-{first_half.index.max().date()} "
          f"vs {second_half.index.min().date()}-{second_half.index.max().date()} ===")

    granger_h1 = granger_edges(first_half, tickers, max_lag=2, alpha=0.05, bonferroni=True)
    granger_h2 = granger_edges(second_half, tickers, max_lag=2, alpha=0.05, bonferroni=True)
    corr_h1 = correlation_edges(first_half, tickers, threshold=0.3)
    corr_h2 = correlation_edges(second_half, tickers, threshold=0.3)

    granger_jaccard = jaccard(set(granger_h1.keys()), set(granger_h2.keys()))
    corr_jaccard = jaccard(set(corr_h1.keys()), set(corr_h2.keys()))

    print(f"Granger edges H1: {len(granger_h1)}, H2: {len(granger_h2)}, Jaccard overlap: {granger_jaccard:.3f}")
    print(f"Correlation edges H1: {len(corr_h1)}, H2: {len(corr_h2)}, Jaccard overlap: {corr_jaccard:.3f}")

    results["stability_check"] = {
        "period_1": f"{first_half.index.min().date()} to {first_half.index.max().date()}",
        "period_2": f"{second_half.index.min().date()} to {second_half.index.max().date()}",
        "granger_h1_edges": len(granger_h1), "granger_h2_edges": len(granger_h2),
        "granger_jaccard_overlap": round(granger_jaccard, 3),
        "correlation_h1_edges": len(corr_h1), "correlation_h2_edges": len(corr_h2),
        "correlation_jaccard_overlap": round(corr_jaccard, 3),
    }

    print("\n=== Full period: Granger causality, UNCORRECTED alpha=0.05 (for comparison) ===")
    granger_uncorrected = granger_edges(returns, tickers, max_lag=2, alpha=0.05, bonferroni=False)
    print(f"Edges found: {len(granger_uncorrected)}  |  sector breakdown: {edge_sector_breakdown(granger_uncorrected, SECTOR)}")
    for (a, b), info in sorted(granger_uncorrected.items(), key=lambda x: x[1]["p_value"]):
        print(f"  {a} -> {b}  (lag={info['lag']}, p={info['p_value']:.5f})")
    results["full_period"]["granger_uncorrected_edge_count"] = len(granger_uncorrected)
    results["full_period"]["granger_uncorrected_edges"] = [
        {"from": a, "to": b, "lag": info["lag"], "p_value": round(info["p_value"], 5)}
        for (a, b), info in granger_uncorrected.items()
    ]

    granger_h1_unc = granger_edges(first_half, tickers, max_lag=2, alpha=0.05, bonferroni=False)
    granger_h2_unc = granger_edges(second_half, tickers, max_lag=2, alpha=0.05, bonferroni=False)
    granger_unc_jaccard = jaccard(set(granger_h1_unc.keys()), set(granger_h2_unc.keys()))
    print(f"Uncorrected Granger edges H1: {len(granger_h1_unc)}, H2: {len(granger_h2_unc)}, "
          f"Jaccard overlap: {granger_unc_jaccard:.3f}")
    common_edges = set(granger_h1_unc.keys()) & set(granger_h2_unc.keys())
    print(f"Edges present in BOTH halves: {common_edges if common_edges else 'none'}")
    results["stability_check"]["granger_uncorrected_h1_edges"] = len(granger_h1_unc)
    results["stability_check"]["granger_uncorrected_h2_edges"] = len(granger_h2_unc)
    results["stability_check"]["granger_uncorrected_jaccard_overlap"] = round(granger_unc_jaccard, 3)
    results["stability_check"]["granger_uncorrected_edges_in_both_halves"] = [f"{a}->{b}" for a, b in common_edges]

    with open("/home/claude/causalyst/data/real_data_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved -> data/real_data_results.json")
    return results


if __name__ == "__main__":
    run()
