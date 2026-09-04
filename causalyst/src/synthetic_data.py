"""
synthetic_data.py

Generates synthetic (but structurally realistic) daily market data with a KNOWN
ground-truth causal structure. This lets us validate the causal-discovery step
(Stage 3) against ground truth BEFORE we plug in real, noisy market data where
the true causal graph is unknown.

Ground truth encoded here (edit freely):
    macro_rate  -> fii_flow        (lag 1 day)
    fii_flow    -> stock_A_return  (lag 1 day)   <- the institutional-flow edge
    dii_flow    -> stock_A_return  (lag 2 days)  <- the institutional-flow edge
    sentiment_A -> stock_A_return  (lag 1 day)
    stock_A_return -> stock_B_return (lag 1 day)  <- sector spillover
    (stock_B has NO direct institutional-flow edge -> used as a negative control)

This is exactly the setup a reviewer will ask for: "how do you know your causal
discovery step actually finds real causes and not spurious correlations?" Answer:
we first show it recovers a known synthetic ground truth, then apply it to real data.
"""

import numpy as np
import pandas as pd


def generate_synthetic_market(n_days: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Exogenous macro driver (e.g., a stand-in for RBI repo rate surprises)
    macro_rate = rng.normal(0, 1, n_days).cumsum() * 0.02

    # FII flow reacts to macro_rate with a 1-day lag, plus noise
    fii_flow = np.zeros(n_days)
    for t in range(1, n_days):
        fii_flow[t] = -0.6 * macro_rate[t - 1] + rng.normal(0, 1)

    # DII flow is structurally independent of macro_rate (SIP-driven, per real
    # Indian market literature showing DII flows are steadier / less macro-reactive)
    dii_flow = rng.normal(0, 1, n_days).cumsum() * 0.01
    dii_flow = np.diff(dii_flow, prepend=dii_flow[0])

    # News sentiment for stock A: exogenous, independent driver
    sentiment_A = rng.normal(0, 1, n_days)

    # Stock A return: caused by FII flow (lag 1), DII flow (lag 2), sentiment (lag 1)
    stock_A_return = np.zeros(n_days)
    for t in range(2, n_days):
        stock_A_return[t] = (
            0.5 * fii_flow[t - 1]
            + 0.3 * dii_flow[t - 2]
            + 0.4 * sentiment_A[t - 1]
            + rng.normal(0, 1) * 0.5
        )

    # Stock B: same sector as A, caused ONLY by A's return (spillover), NOT by
    # institutional flow directly. This is the negative control -- a correct
    # causal-discovery method should NOT find a direct FII/DII -> B edge.
    stock_B_return = np.zeros(n_days)
    for t in range(1, n_days):
        stock_B_return[t] = 0.45 * stock_A_return[t - 1] + rng.normal(0, 1) * 0.6

    dates = pd.date_range("2019-01-01", periods=n_days, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "macro_rate": macro_rate,
            "fii_flow": fii_flow,
            "dii_flow": dii_flow,
            "sentiment_A": sentiment_A,
            "stock_A_return": stock_A_return,
            "stock_B_return": stock_B_return,
        }
    )
    return df


GROUND_TRUTH_EDGES = {
    ("macro_rate", "fii_flow"): 1,
    ("fii_flow", "stock_A_return"): 1,
    ("dii_flow", "stock_A_return"): 2,
    ("sentiment_A", "stock_A_return"): 1,
    ("stock_A_return", "stock_B_return"): 1,
}

# Edges that should NOT be found (used to compute false-positive rate)
NEGATIVE_CONTROLS = [
    ("fii_flow", "stock_B_return"),
    ("dii_flow", "stock_B_return"),
    ("sentiment_A", "stock_B_return"),
]

if __name__ == "__main__":
    df = generate_synthetic_market()
    df.to_csv("/home/claude/causalyst/data/synthetic_market.csv", index=False)
    print(df.head())
    print(f"\nGenerated {len(df)} rows -> data/synthetic_market.csv")
