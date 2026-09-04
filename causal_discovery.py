from itertools import permutations
from typing import Dict, List, Tuple
import contextlib
import io

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def granger_edges(df: pd.DataFrame, columns: List[str], max_lag: int = 2,
                   alpha: float = 0.05, bonferroni: bool = True) -> Dict[Tuple[str, str], Dict]:
    edges = {}
    n_pairs = len(columns) * (len(columns) - 1)
    corrected_alpha = alpha / n_pairs if bonferroni else alpha
    for a, b in permutations(columns, 2):
        sub = df[[b, a]].dropna()
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
