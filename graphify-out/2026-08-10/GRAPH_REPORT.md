# Graph Report - causalyst  (2026-08-08)

## Corpus Check
- 18 files · ~20,623 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 120 nodes · 217 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- run_clustering.py
- MarketGraph
- Any
- real_data_experiment.py
- demo.py
- src/causal_discovery.py
- main
- Causalyst — working code scaffold
- agents.py
- DataFrame
- fii_dii_scraper_TEMPLATE.py
- make_equations.py
- make_causal_heatmap.py

## God Nodes (most connected - your core abstractions)
1. `main()` - 16 edges
2. `write_html_report()` - 10 edges
3. `MarketGraph` - 9 edges
4. `load_or_train_phase_a()` - 9 edges
5. `clean_error()` - 8 edges
6. `granger_edges()` - 7 edges
7. `build_undirected_weighted_graph()` - 7 edges
8. `load_returns()` - 6 edges
9. `run()` - 6 edges
10. `discover_edges_from_full_history()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `granger_edges()`  [EXTRACTED]
  causalyst/src/real_data_experiment.py → causal_discovery.py
- `plot_graph()` --calls--> `build_undirected_weighted_graph()`  [EXTRACTED]
  visualize_clusters.py → graph_clustering.py
- `run()` --calls--> `correlation_edges()`  [INFERRED]
  causalyst/src/real_data_experiment.py → causalyst/src/causal_discovery.py
- `run()` --calls--> `edge_sector_breakdown()`  [EXTRACTED]
  causalyst/src/pcmci_discovery.py → causalyst/src/real_data_experiment.py
- `run()` --calls--> `jaccard()`  [EXTRACTED]
  causalyst/src/pcmci_discovery.py → causalyst/src/real_data_experiment.py

## Import Cycles
- None detected.

## Communities (13 total, 2 thin omitted)

### Community 0 - "run_clustering.py"
Cohesion: 0.18
Nodes (16): granger_edges(), DataFrame, Graph, build_undirected_weighted_graph(), louvain_clusters(), modularity_score(), graph_clustering.py The piece that was NOT built before: formal community…, Symmetrize the directed causal graph into an undirected weighted graph suitable… (+8 more)

### Community 1 - "MarketGraph"
Cohesion: 0.15
Nodes (8): build_graph_from_discovery(), MarketGraph, graph_builder.py Stage 2 of the pipeline: build the dynamic market graph. Key…, Everything that Stage 3 found to causally precede `node` -- this is exactly…, Assemble a MarketGraph from the output of causal_discovery.granger_edges()., generate_synthetic_market(), DataFrame, synthetic_data.py Generates synthetic (but structurally realistic) daily market…

### Community 2 - "Any"
Cohesion: 0.28
Nodes (13): Any, clean_error(), compute_clusters(), discover_edges_from_full_history(), edge_weight(), format_parent_evidence(), load_or_train_phase_a(), make_cluster_graph_svg() (+5 more)

### Community 3 - "real_data_experiment.py"
Cohesion: 0.32
Nodes (10): pcmci_discovery.py Runs PCMCI (Peter-Clark Momentary Conditional Independence),…, run(), run_pcmci(), edge_sector_breakdown(), jaccard(), load_returns(), DataFrame, real_data_experiment.py PRELIMINARY REAL-DATA CASE STUDY. Uses REAL daily OHLCV… (+2 more)

### Community 4 - "demo.py"
Cohesion: 0.33
Nodes (10): build_agent_evidence(), cluster_for_stock(), extract_final_confidence(), make_price_chart_svg(), parse_jsonish(), pct_move(), demo.py Live-safe Causalyst demo. Phase A trains or loads a cached causal graph…, render_transcript_html() (+2 more)

### Community 5 - "src/causal_discovery.py"
Cohesion: 0.25
Nodes (8): correlation_edges(), evaluate_against_ground_truth(), granger_edges(), DataFrame, causal_discovery.py Stage 3 of the pipeline: separate real cause-and-effect…, Naive correlation-only baseline. Symmetric -- cannot assign direction, so we…, Pairwise Granger causality: does past `a` help predict `b` beyond b's own past?…, Precision / recall / F1 of discovered edges vs known ground truth, plus a…

### Community 6 - "main"
Cohesion: 0.32
Nodes (6): build_project_graph(), cached_debate_transcript(), DemoGraph, main(), run_debate_safely(), section()

### Community 7 - "Causalyst — working code scaffold"
Cohesion: 0.29
Nodes (6): Causalyst — working code scaffold, Causalyst — working code scaffold, What is genuinely NOT done — do not present these as finished, What needs your API keys / accounts, What's REAL and already run (not synthetic, not fabricated), What's real and runs on synthetic data (ground-truth validated)

### Community 8 - "agents.py"
Cohesion: 0.40
Nodes (5): AgentResult, _call_claude(), agents.py Stage 4: multi-agent debate and synthesis. IMPORTANT design note…, evidence: {agent_role: [list of evidence strings relevant to that agent]} e.g.…, run_debate()

### Community 9 - "DataFrame"
Cohesion: 0.60
Nodes (6): load_cached_prices(), load_phase_b_prices(), load_training_returns(), normalize_live_price_payload(), DataFrame, try_upstox_live_prices()

### Community 10 - "fii_dii_scraper_TEMPLATE.py"
Cohesion: 0.40
Nodes (3): fetch_today(), fii_dii_scraper_TEMPLATE.py HONESTY NOTE: this script could NOT be tested or…, UNVERIFIED -- NSE's site requires session cookies/headers that change; you will…

## Knowledge Gaps
- **6 isolated node(s):** `AgentResult`, `Causalyst — working code scaffold`, `What's REAL and already run (not synthetic, not fabricated)`, `What's real and runs on synthetic data (ground-truth validated)`, `What needs your API keys / accounts` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `granger_edges()` connect `run_clustering.py` to `MarketGraph`, `real_data_experiment.py`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `run()` connect `real_data_experiment.py` to `run_clustering.py`, `src/causal_discovery.py`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **What connects `AgentResult`, `Causalyst — working code scaffold`, `What's REAL and already run (not synthetic, not fabricated)` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MarketGraph` be split into smaller, more focused modules?**
  _Cohesion score 0.14705882352941177 - nodes in this community are weakly interconnected._