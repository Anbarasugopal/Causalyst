"""
causal_discovery.py

Stage 3 of the pipeline: separate real cause-and-effect from mere correlation.

Implements:
  - correlation_edges(): naive baseline -- just thresholds Pearson correlation
  - granger_edges(): pairwise Granger causality test (statsmodels), sweeping
    lags 1..max_lag and keeping the most significant lag per pair
  - evaluate_against_ground_truth(): precision/recall/F1 against a known
    ground-truth edge set (used on synthetic data to validate the method
    before trusting it on real, unlabeled market data)

Note on method choice for the paper: PCMCI (via `tigramite`) is the more
rigorous multivariate causal-discovery method referenced in the project
proposal, and controls for confounding across all variables jointly rather
than testing pairs in isolation. This module ships pairwise Granger causality
as a lightweight, dependency-light baseline that runs anywhere; swap in
tigramite's PCMCI for the paper's main reported results once installed
(`pip install tigramite`), keeping this module's interface unchanged so the
rest of the pipeline (graph builder, GNN, agents) doesn't need to change.
"""

from itertools import permutations
from typing import Dict, List, Tuple
import contextlib
import io

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def correlation_edges(df: pd.DataFrame, columns: List[str], threshold: float = 0.3) -> Dict[Tuple[str, str], float]:
    """Naive correlation-only baseline. Symmetric -- cannot assign direction,
    so we report both directions with the same score to mirror how a
    correlation-only system would (incorrectly) treat the relationship."""
    edges = {}
    corr = df[columns].corr()
    for a, b in permutations(columns, 2):
        score = abs(corr.loc[a, b])
        if score >= threshold:
            edges[(a, b)] = score
    return edges


def granger_edges(
    df: pd.DataFrame,
    columns: List[str],
    max_lag: int = 3,
    alpha: float = 0.05,
    bonferroni: bool = True,
) -> Dict[Tuple[str, str], Dict]:
    """
    Pairwise Granger causality: does past `a` help predict `b` beyond b's own
    past? Tests a -> b for every ordered pair, sweeps lags 1..max_lag, and
    keeps the lag with the lowest p-value if it clears `alpha`.

    Returns {(a, b): {"lag": int, "p_value": float, "f_stat": float}}
    """
    edges = {}
    n_pairs = len(columns) * (len(columns) - 1)
    corrected_alpha = alpha / n_pairs if bonferroni else alpha
    for a, b in permutations(columns, 2):
        sub = df[[b, a]].dropna()  # statsmodels expects [target, predictor]
        best = None
        for lag in range(1, max_lag + 1):
            if len(sub) <= lag * 4:
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = grangercausalitytests(sub, maxlag=[lag])
                f_test = result[lag][0]["ssr_ftest"]
                f_stat, p_value = f_test[0], f_test[1]
            except Exception:
                continue
            if best is None or p_value < best["p_value"]:
                best = {"lag": lag, "p_value": p_value, "f_stat": f_stat}
        if best is not None and best["p_value"] < corrected_alpha:
            edges[(a, b)] = best
    return edges


def evaluate_against_ground_truth(
    discovered: Dict[Tuple[str, str], object],
    ground_truth: Dict[Tuple[str, str], int],
    negative_controls: List[Tuple[str, str]],
) -> Dict[str, float]:
    """Precision / recall / F1 of discovered edges vs known ground truth,
    plus a false-positive count on edges that should NOT exist (negative
    controls). This is the validation you report in the paper BEFORE
    running the method on real, unlabeled market data."""
    discovered_set = set(discovered.keys())
    truth_set = set(ground_truth.keys())

    tp = len(discovered_set & truth_set)
    fp = len(discovered_set - truth_set)
    fn = len(truth_set - discovered_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    fp_on_negative_controls = sum(1 for edge in negative_controls if edge in discovered_set)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positives_on_negative_controls": fp_on_negative_controls,
        "negative_controls_tested": len(negative_controls),
    }


if __name__ == "__main__":
    from synthetic_data import GROUND_TRUTH_EDGES, NEGATIVE_CONTROLS, generate_synthetic_market

    df = generate_synthetic_market()
    cols = ["macro_rate", "fii_flow", "dii_flow", "sentiment_A", "stock_A_return", "stock_B_return"]

    print("=== Correlation-only baseline ===")
    corr = correlation_edges(df, cols, threshold=0.15)
    corr_eval = evaluate_against_ground_truth(corr, GROUND_TRUTH_EDGES, NEGATIVE_CONTROLS)
    print(f"Edges found: {len(corr)}")
    print(corr_eval)

    print("\n=== Granger causality (our method) ===")
    granger = granger_edges(df, cols, max_lag=3)
    granger_eval = evaluate_against_ground_truth(granger, GROUND_TRUTH_EDGES, NEGATIVE_CONTROLS)
    for edge, info in granger.items():
        print(f"  {edge}: lag={info['lag']}, p={info['p_value']:.4f}")
    print(granger_eval)
