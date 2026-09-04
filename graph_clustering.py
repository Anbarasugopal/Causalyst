"""
graph_clustering.py

The piece that was NOT built before: formal community detection on top of
the causal market graph, rather than a human eyeballing a heatmap.

Takes the directed causal graph (from causal_discovery.py's output, the same
structure graph_builder.py's MarketGraph wraps) and partitions it into
clusters using two methods:

  1. weakly_connected_clusters() -- the simplest possible notion of a
     "cluster": stocks connected by ANY chain of causal edges, ignoring
     direction and strength. Good as a sanity-check baseline.

  2. louvain_clusters() -- proper community detection via modularity
     maximization (Louvain algorithm, via networkx's built-in implementation,
     no extra package needed). This is what should actually be called
     "graphical clusters" -- it finds groups of stocks that are more densely
     interconnected with each other than with the rest of the graph, which
     is a meaningfully different (and stronger) claim than "share a sector
     label" or "lit up together on a heatmap I looked at."

Both operate on an undirected, weighted symmetrization of the causal graph:
edge weight = -log10(p_value), so stronger (more significant) causal links
pull nodes together more strongly in the clustering than weak ones.
"""

from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np


def build_undirected_weighted_graph(causal_edges: Dict[Tuple[str, str], Dict]) -> nx.Graph:
    """Symmetrize the directed causal graph into an undirected weighted graph
    suitable for community detection. If both A->B and B->A exist, their
    weights are summed (both directions of causal interconnection count
    toward keeping A and B in the same cluster)."""
    g = nx.Graph()
    for (src, tgt), info in causal_edges.items():
        w = -np.log10(max(info["p_value"], 1e-12))
        if g.has_edge(src, tgt):
            g[src][tgt]["weight"] += w
        else:
            g.add_edge(src, tgt, weight=w)
    return g


def weakly_connected_clusters(causal_edges: Dict[Tuple[str, str], Dict]) -> List[Set[str]]:
    """Baseline: stocks reachable from each other via any causal edge,
    ignoring direction and strength entirely."""
    g = nx.DiGraph()
    g.add_edges_from(causal_edges.keys())
    return [set(c) for c in nx.weakly_connected_components(g)]


def louvain_clusters(causal_edges: Dict[Tuple[str, str], Dict], seed: int = 42) -> List[Set[str]]:
    """Proper community detection via modularity maximization (Louvain)."""
    g = build_undirected_weighted_graph(causal_edges)
    if g.number_of_edges() == 0:
        return [{n} for n in g.nodes()]
    communities = nx.community.louvain_communities(g, weight="weight", seed=seed)
    return [set(c) for c in communities]


def modularity_score(causal_edges: Dict[Tuple[str, str], Dict], clusters: List[Set[str]]) -> float:
    """How much better than random this partition is (0 = no better than
    random, higher = stronger real cluster structure). Standard way to
    report whether the clusters found are actually meaningful."""
    g = build_undirected_weighted_graph(causal_edges)
    if g.number_of_edges() == 0:
        return 0.0
    return nx.community.modularity(g, clusters, weight="weight")


def summarize_clusters(clusters: List[Set[str]], sector_map: Dict[str, str] = None) -> None:
    for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)):
        if len(cluster) <= 1:
            continue
        if sector_map:
            sectors = [sector_map.get(t, "?") for t in cluster]
            purity = max(sectors.count(s) for s in set(sectors)) / len(sectors)
            print(f"Cluster {i} ({len(cluster)} stocks, {purity:.0%} same-sector): {sorted(cluster)}")
        else:
            print(f"Cluster {i} ({len(cluster)} stocks): {sorted(cluster)}")
