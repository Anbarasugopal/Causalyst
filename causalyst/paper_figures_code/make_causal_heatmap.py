"""
make_causal_heatmap.py

Generates a REAL causality-matrix heatmap from the actual PCMCI results
(data/pcmci_results.json) on the 10 real NSE stocks -- the direct analogue
of MagicNet's Fig. 6 causality-matrix visualization. Every value plotted
here came from an actual PCMCI run on actual NSE price data; nothing here
is illustrative or fabricated.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "KOTAKBANK",
           "ICICIBANK", "SBIN", "HCLTECH", "WIPRO", "AXISBANK"]
SECTOR = {
    "RELIANCE": "Energy", "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "HDFCBANK": "Bank", "ICICIBANK": "Bank", "KOTAKBANK": "Bank",
    "AXISBANK": "Bank", "SBIN": "Bank",
}
# Order stocks by sector so the block structure is visible, like MagicNet's Fig. 6
ORDER = ["RELIANCE", "TCS", "INFY", "HCLTECH", "WIPRO",
         "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN"]

with open("/home/claude/causalyst/data/pcmci_results.json") as f:
    pcmci = json.load(f)

n = len(ORDER)
idx = {t: i for i, t in enumerate(ORDER)}
# strength matrix = -log10(p_value) for found edges, 0 otherwise (higher = stronger evidence)
mat = np.zeros((n, n))
for e in pcmci["full_period_edges"]:
    i, j = idx[e["from"]], idx[e["to"]]
    strength = -np.log10(max(e["p_value"], 1e-10))
    mat[i, j] = strength

fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=200)
im = ax.imshow(mat, cmap="Reds", vmin=0, aspect="equal")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(ORDER, rotation=90, fontsize=8)
ax.set_yticklabels(ORDER, fontsize=8)
ax.set_xlabel("Effect (caused)", fontsize=9)
ax.set_ylabel("Cause (source)", fontsize=9)

# sector divider lines (IT block = first 5, Bank block = last 5)
ax.axhline(4.5, color="black", linewidth=0.8)
ax.axvline(4.5, color="black", linewidth=0.8)
ax.set_title("Real PCMCI-Discovered Causal Matrix\n10 NSE stocks, 2012\u20132021, $\\tau_{max}=2$", fontsize=10, pad=34)
ax.text(2, -2.6, "IT", ha="center", fontsize=9, fontweight="bold")
ax.text(7, -2.6, "Banking", ha="center", fontsize=9, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$-\log_{10}(p)$  (PCMCI significance; 0 = no link found)", fontsize=8)

plt.tight_layout()
plt.savefig("/home/claude/causalyst/paper/figures/causal_heatmap.png", dpi=200, bbox_inches="tight")
print("saved causal_heatmap.png")
print(f"Non-zero entries: {int(np.count_nonzero(mat))} / {n*n - n} possible directed pairs")
