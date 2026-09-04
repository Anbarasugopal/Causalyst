import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


def run_pcmci(returns_df, tau_max=2, pc_alpha=0.05):
    var_names = list(returns_df.columns)
    dataframe = pp.DataFrame(returns_df.values, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    p_matrix = results["p_matrix"]
    edges = {}
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for tau in range(1, tau_max + 1):
                p = p_matrix[i, j, tau]
                if p < pc_alpha:
                    key = (var_names[i], var_names[j])
                    if key not in edges or p < edges[key]["p_value"]:
                        edges[key] = {"lag": tau, "p_value": float(p)}
    return edges
