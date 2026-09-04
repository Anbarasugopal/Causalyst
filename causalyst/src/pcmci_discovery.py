"""
pcmci_discovery.py

Runs PCMCI (Peter-Clark Momentary Conditional Independence), a multivariate
constraint-based causal discovery method, on the SAME real 10-stock NSE
return data used in real_data_experiment.py. This directly addresses the
limitation flagged throughout this project: pairwise Granger causality does
not control for confounding across the full variable set, while PCMCI tests
each candidate link conditional on the other variables, which should reduce
the false-positive-from-indirect-paths problem visible in the pairwise
Granger results.

This is a genuine multivariate re-analysis of real market data, not a
simulation -- report whatever it finds, including if it finds LESS than
pairwise Granger (that would itself support the confounding concern) or
a different set of edges (would suggest some pairwise "discoveries" were
in fact indirect/confounded).
"""

import json

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from real_data_experiment import SECTOR, edge_sector_breakdown, jaccard, load_returns


def run_pcmci(returns_df, tau_max=2, pc_alpha=0.05):
    var_names = list(returns_df.columns)
    data = returns_df.values
    dataframe = pp.DataFrame(data, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    p_matrix = results["p_matrix"]  # shape (n_vars, n_vars, tau_max+1); [i, j, tau] = link i -> j at lag tau
    edges = {}
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for tau in range(1, tau_max + 1):  # skip tau=0 (contemporaneous) -- we want directed, lagged links
                p = p_matrix[i, j, tau]
                if p < pc_alpha:
                    key = (var_names[i], var_names[j])
                    if key not in edges or p < edges[key]["p_value"]:
                        edges[key] = {"lag": tau, "p_value": float(p)}
    return edges


def run():
    returns = load_returns()
    tickers = list(returns.columns)
    n_days = len(returns)

    print(f"Running PCMCI on REAL data: {n_days} days, {len(tickers)} stocks\n")

    print("=== Full period: PCMCI (tau_max=2, pc_alpha=0.05) ===")
    pcmci_full = run_pcmci(returns, tau_max=2, pc_alpha=0.05)
    print(f"Edges found: {len(pcmci_full)}  |  sector breakdown: {edge_sector_breakdown(pcmci_full, SECTOR)}")
    for (a, b), info in sorted(pcmci_full.items(), key=lambda x: x[1]["p_value"]):
        print(f"  {a} -> {b}  (lag={info['lag']}, p={info['p_value']:.5f})")

    midpoint = returns.index[n_days // 2]
    first_half = returns[returns.index < midpoint]
    second_half = returns[returns.index >= midpoint]

    print(f"\n=== Stability check across the same two 5-year halves ===")
    pcmci_h1 = run_pcmci(first_half, tau_max=2, pc_alpha=0.05)
    pcmci_h2 = run_pcmci(second_half, tau_max=2, pc_alpha=0.05)
    overlap = jaccard(set(pcmci_h1.keys()), set(pcmci_h2.keys()))
    common = set(pcmci_h1.keys()) & set(pcmci_h2.keys())
    print(f"PCMCI edges H1: {len(pcmci_h1)}, H2: {len(pcmci_h2)}, Jaccard overlap: {overlap:.3f}")
    print(f"Edges present in BOTH halves: {common if common else 'none'}")

    out = {
        "full_period_edge_count": len(pcmci_full),
        "full_period_sector_breakdown": edge_sector_breakdown(pcmci_full, SECTOR),
        "full_period_edges": [
            {"from": a, "to": b, "lag": info["lag"], "p_value": round(info["p_value"], 5)}
            for (a, b), info in pcmci_full.items()
        ],
        "h1_edge_count": len(pcmci_h1),
        "h2_edge_count": len(pcmci_h2),
        "jaccard_overlap": round(overlap, 3),
        "edges_in_both_halves": [f"{a}->{b}" for a, b in common],
    }
    with open("/home/claude/causalyst/data/pcmci_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved -> data/pcmci_results.json")
    return out


if __name__ == "__main__":
    run()
