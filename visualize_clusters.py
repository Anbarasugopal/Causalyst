import sys
sys.path.insert(0, "/home/claude/cluster")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from causal_discovery import granger_edges
from pcmci_discovery import run_pcmci
from graph_clustering import build_undirected_weighted_graph, louvain_clusters, modularity_score
from run_clustering import load_returns, SECTOR

CLUSTER_COLORS = ["#4BACC6", "#F79646", "#9BBB59", "#C0504D", "#8064A2"]
SECTOR_SHAPE = {"IT": "o", "BANKING": "s", "ENERGY": "^"}

def plot_graph(edges, clusters, title, ax):
    g = build_undirected_weighted_graph(edges)
    for n in SECTOR:
        if n not in g:
            g.add_node(n)
    pos = nx.spring_layout(g, seed=7, k=1.1)

    node_cluster = {}
    for i, c in enumerate(clusters):
        for n in c:
            node_cluster[n] = i

    for shape_sector, marker in SECTOR_SHAPE.items():
        nodes = [n for n in g.nodes() if SECTOR.get(n) == shape_sector]
        colors = [CLUSTER_COLORS[node_cluster.get(n, 0) % len(CLUSTER_COLORS)] for n in nodes]
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors, node_shape=marker,
                                node_size=1400, edgecolors="black", linewidths=1.3, ax=ax)

    weights = [g[u][v]["weight"] for u, v in g.edges()]
    max_w = max(weights) if weights else 1
    nx.draw_networkx_edges(g, pos, width=[1 + 3*w/max_w for w in weights], edge_color="#999999",
                            alpha=0.6, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=9, font_weight="bold", ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#002060")
    ax.axis("off")

returns = load_returns()
tickers = list(returns.columns)

granger_e = granger_edges(returns, tickers, max_lag=2, alpha=0.05, bonferroni=False)
granger_c = louvain_clusters(granger_e)
granger_mod = modularity_score(granger_e, granger_c)

pcmci_e = run_pcmci(returns, tau_max=2, pc_alpha=0.05)
pcmci_c = louvain_clusters(pcmci_e)
pcmci_mod = modularity_score(pcmci_e, pcmci_c)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=180)
plot_graph(granger_e, granger_c, f"Granger causal graph, Louvain clusters\n(color = cluster, shape = true sector, modularity = {granger_mod:.3f})", axes[0])
plot_graph(pcmci_e, pcmci_c, f"PCMCI causal graph, Louvain clusters\n(color = cluster, shape = true sector, modularity = {pcmci_mod:.3f})", axes[1])

circle_leg = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='IT'),
              plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='gray', markersize=12, label='Banking'),
              plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='gray', markersize=12, label='Energy')]
fig.legend(handles=circle_leg, loc='lower center', ncol=3, frameon=False, fontsize=11)

plt.suptitle("If clustering matched sector, node SHAPE and node COLOR would align.\nOn this 10-stock real dataset they mostly don't -- modularity is weak both ways.",
             fontsize=11.5, y=1.02, style="italic", color="#555")
plt.tight_layout()
plt.savefig("/home/claude/cluster/cluster_comparison.png", dpi=180, bbox_inches="tight")
print("saved cluster_comparison.png")
