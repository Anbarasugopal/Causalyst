# Graph Report - causalyst  (2026-08-13)

## Corpus Check
- 27 files · ~30,782 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 204 nodes · 343 edges · 15 communities (12 shown, 3 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- run_clustering.py
- MarketGraph
- demo.py
- real_data_experiment.py
- evaluate_models.py
- src/causal_discovery.py
- api.py
- Causalyst — working code scaffold
- agents.py
- Causalyst — Project Handoff Brief
- fii_dii_scraper_TEMPLATE.py
- make_equations.py
- make_causal_heatmap.py
- gnn_prototype.py

## God Nodes (most connected - your core abstractions)
1. `main()` - 16 edges
2. `write_html_report()` - 10 edges
3. `MarketGraph` - 9 edges
4. `load_or_train_phase_a()` - 9 edges
5. `train_one_model()` - 9 edges
6. `granger_edges()` - 8 edges
7. `clean_error()` - 8 edges
8. `load_assets()` - 7 edges
9. `StockWindowDataset` - 7 edges
10. `evaluate_at_threshold()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `predict()` --indirect_call--> `edge_weight()`  [INFERRED]
  api.py → demo.py
- `run()` --calls--> `granger_edges()`  [EXTRACTED]
  causalyst/src/real_data_experiment.py → causal_discovery.py
- `plot_graph()` --calls--> `build_undirected_weighted_graph()`  [EXTRACTED]
  visualize_clusters.py → graph_clustering.py
- `try_live_full_graph_state()` --calls--> `load_saved_token()`  [EXTRACTED]
  api.py → upstox_auth.py
- `try_live_full_graph_state()` --calls--> `fetch_daily_candles()`  [EXTRACTED]
  api.py → upstox_fetch.py

## Import Cycles
- None detected.

## Communities (15 total, 3 thin omitted)

### Community 0 - "run_clustering.py"
Cohesion: 0.16
Nodes (17): granger_edges(), DataFrame, graph_builder.py Stage 2 of the pipeline: build the dynamic market graph. Key…, Graph, build_undirected_weighted_graph(), louvain_clusters(), modularity_score(), graph_clustering.py The piece that was NOT built before: formal community… (+9 more)

### Community 1 - "MarketGraph"
Cohesion: 0.22
Nodes (4): build_graph_from_discovery(), MarketGraph, Everything that Stage 3 found to causally precede `node` -- this is exactly…, Assemble a MarketGraph from the output of causal_discovery.granger_edges().

### Community 2 - "demo.py"
Cohesion: 0.15
Nodes (35): Any, build_agent_evidence(), build_project_graph(), cached_debate_transcript(), clean_error(), cluster_for_stock(), compute_clusters(), DemoGraph (+27 more)

### Community 3 - "real_data_experiment.py"
Cohesion: 0.32
Nodes (10): pcmci_discovery.py Runs PCMCI (Peter-Clark Momentary Conditional Independence),…, run(), run_pcmci(), edge_sector_breakdown(), jaccard(), load_returns(), DataFrame, real_data_experiment.py PRELIMINARY REAL-DATA CASE STUDY. Uses REAL daily OHLCV… (+2 more)

### Community 4 - "evaluate_models.py"
Cohesion: 0.10
Nodes (27): Dataset, CausalTemporalGNN, collect_logits_and_labels(), evaluate_at_threshold(), load_data(), load_graphs(), main(), evaluate_models.py Comprehensive evaluation for Causalyst's three model… (+19 more)

### Community 5 - "src/causal_discovery.py"
Cohesion: 0.18
Nodes (11): correlation_edges(), evaluate_against_ground_truth(), granger_edges(), DataFrame, causal_discovery.py Stage 3 of the pipeline: separate real cause-and-effect…, Naive correlation-only baseline. Symmetric -- cannot assign direction, so we…, Pairwise Granger causality: does past `a` help predict `b` beyond b's own past?…, Precision / recall / F1 of discovered edges vs known ground truth, plus a… (+3 more)

### Community 6 - "api.py"
Cohesion: 0.11
Nodes (24): build_clusters(), causal_model_metrics(), CausalTemporalGNN, cluster_payload(), index(), load_assets(), load_cached_state(), predict() (+16 more)

### Community 7 - "Causalyst — working code scaffold"
Cohesion: 0.29
Nodes (6): Causalyst — working code scaffold, Causalyst — working code scaffold, What is genuinely NOT done — do not present these as finished, What needs your API keys / accounts, What's REAL and already run (not synthetic, not fabricated), What's real and runs on synthetic data (ground-truth validated)

### Community 8 - "agents.py"
Cohesion: 0.40
Nodes (5): AgentResult, _call_claude(), agents.py Stage 4: multi-agent debate and synthesis. IMPORTANT design note…, evidence: {agent_role: [list of evidence strings relevant to that agent]} e.g.…, run_debate()

### Community 9 - "Causalyst — Project Handoff Brief"
Cohesion: 0.25
Nodes (7): Causalyst — Project Handoff Brief, Current status — what's real and validated, what isn't, Do not violate these — each cost real effort to establish, Files that exist (read them, don't rebuild from scratch), Immediate next task (in priority order), Security note, What this project is

### Community 10 - "fii_dii_scraper_TEMPLATE.py"
Cohesion: 0.40
Nodes (3): fetch_today(), fii_dii_scraper_TEMPLATE.py HONESTY NOTE: this script could NOT be tested or…, UNVERIFIED -- NSE's site requires session cookies/headers that change; you will…

## Knowledge Gaps
- **12 isolated node(s):** `AgentResult`, `What this project is`, `Current status — what's real and validated, what isn't`, `Do not violate these — each cost real effort to establish`, `Files that exist (read them, don't rebuild from scratch)` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `predict()` connect `api.py` to `demo.py`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Why does `edge_weight()` connect `demo.py` to `api.py`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **What connects `AgentResult`, `What this project is`, `Current status — what's real and validated, what isn't` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `demo.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14509246088193456 - nodes in this community are weakly interconnected._
- **Should `evaluate_models.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0962566844919786 - nodes in this community are weakly interconnected._
- **Should `api.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10685483870967742 - nodes in this community are weakly interconnected._