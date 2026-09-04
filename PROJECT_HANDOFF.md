# Causalyst — Project Handoff Brief

Give this whole document to the new agent as its first message, before any task
instruction. It has no memory of prior work — everything below is real, hard-won
context from building this, not guesses. Do not let it silently re-derive or
override any of the "Do not violate" items.

## What this project is

A causal, multi-agent AI system for explainable Indian stock-market movement
prediction. Core claim: model relationships between stocks (and institutional
FII/DII flow) as *causally discovered* graph structure, not raw correlation,
then explain predictions via a structured multi-agent LLM debate rather than a
black-box signal.

## Current status — what's real and validated, what isn't

| Component | Status |
|---|---|
| Causal discovery (Granger + PCMCI) | Validated on synthetic ground truth AND real 10-stock NSE data |
| Graph clustering (Louvain) | Built and tested; results are **weak** (modularity 0.06–0.12), don't oversell |
| Temporal GNN architecture | Validated end-to-end at small scale, then trained at real 376-stock scale on Colab |
| **Real Colab training results** | 376 stocks retained (of Nifty 500 — see note below), 26,959 causal edges, 5 Louvain clusters, modularity 0.0866. Causal-graph GNN: test F1 0.3789. GRU-only and correlation-graph baselines both **collapsed** to near-zero F1 (~0, 0.0002) despite similar ~51% accuracy — accuracy is not meaningful here since the test set is ~49% positive. |
| Threshold tuning / precision-recall breakdown | **Not yet done** — was the agreed next step before this handoff. Do this before trusting the F1 numbers further; a model can hit F1=0.38 via poor precision just as easily as real skill. |
| Multi-seed stability check | **Not done.** One run, one seed. Whether the causal model's advantage is real or a lucky init is unknown. |
| Live demo (demo.py) | Built, live-safe, falls back cleanly if APIs/keys are missing |
| Frontend (stock_prediction_web.html) | Built, currently **mock data only**, explicitly labeled "Demo estimate" |
| Backend API wiring | **Not done — this is the actual next task**, see below |
| Upstox live data | Auth flow built and working. **Only 10 of 376 stocks have mapped instrument_keys** — see constraint below |

## Do not violate these — each cost real effort to establish

1. **The GNN is a graph model.** Predicting even one stock requires running the
   *entire* 376-stock graph through the GCN layers (message passing needs every
   node's features), then reading off the one requested ticker. Never run
   inference on a single stock in isolation.

2. **Never re-run causal discovery (Granger/PCMCI) on short windows.** 5 days is
   nowhere near enough data — the code's own lag-length checks will reject it,
   and even if forced through, the result is statistically meaningless. The
   causal graph is built once from full history (2012–2021) and cached; it is
   NOT rebuilt per-prediction or per-demo-run.

3. **Bonferroni correction was rejected in favor of Benjamini-Hochberg FDR** for
   the 500-stock-scale graph. At ~250K pairwise comparisons, Bonferroni is so
   strict it produces a near-empty graph (confirmed: it left only 1/90 edges
   even at just 10 stocks). Don't switch back to Bonferroni "for rigor" without
   understanding this tradeoff was deliberate.

4. **"376 stocks," never "full Nifty 500" or "all Indian stocks."** Only 376 of
   500 survived a data-coverage filter (`dropna(thresh=80%)` then intersect all
   dates), and that filter is NOT random — it systematically favors stocks with
   the longest, most complete listing history, i.e. skews toward large/established
   names. State this precisely in any report or UI text.

5. **Modularity did NOT improve going from 10 to 376 stocks** (0.122 → 0.0866 for
   Granger-based graphs) — the naive "more data = cleaner clusters" assumption is
   wrong here. Report this plainly if asked about clustering quality.

6. **The model predicts 1 trading day ahead only.** It has no native multi-day
   capability. Multi-day horizons (5D/20D/60D/custom) use naive iterative
   chaining (feed day+1's prediction back in as day+2's input) — this MUST be
   visibly labeled as a heuristic in any UI, with confidence shown declining as
   horizon increases. Never present a chained multi-day output with the same
   confidence framing as the real 1-day prediction.

7. **Live Upstox data currently covers only 10 of the 376 stocks** in the trained
   graph (instrument_key mapping was only done for the original demo set).
   Full-graph live data is a real, separate mapping task, not a quick fix.
   Current design: serve predictions from cached `returns.parquet` (all 376
   stocks, but stale — show the real last-available date honestly in the UI),
   with an optional `/api/refresh_live` endpoint that overlays live data for
   just the one queried ticker. Do not attempt full 376-stock live fetching
   without treating it as its own scoped task.

8. **Any confidence/accuracy/threshold number shown anywhere must come from a
   real evaluation run** (stored in `metadata.json` once threshold-tuning is
   done) — never a hardcoded placeholder or invented figure.

9. **Never claim the demo webpage shows real predictions until the backend is
   actually wired and tested.** It was explicitly labeled "Demo estimate" for a
   reason — hold that line until `/api/predict` is real and verified.

## Files that exist (read them, don't rebuild from scratch)

`synthetic_data.py`, `causal_discovery.py`, `pcmci_discovery.py`, `graph_builder.py`,
`graph_clustering.py`, `run_clustering.py`, `visualize_clusters.py`, `agents.py`,
`gnn_prototype.py` (validated small-scale architecture — the 376-stock Colab
model extends this same architecture, don't redesign it), `demo.py`,
`demo_report.html`, `stock_prediction_web.html`, `upstox_auth.py`, `upstox_fetch.py`
(only 10 stocks mapped — see constraint 7), `causalyst_nifty500_colab_gnn.ipynb`.

Backend artifacts (from Colab, should be in the project folder):
`model_checkpoint_graph.pt`, `causal_graph.pt`, `returns.parquet`, `metadata.json`.

## Security note

API credentials were briefly exposed in an earlier chat session and should be
treated as rotated/invalid. Confirm a fresh `.env` (not `.env.example`) exists
locally with a newly-regenerated Upstox API secret before running anything that
touches live data. Never let credentials appear in chat, logs, or committed files.

## Immediate next task (in priority order)

1. **Precision/recall breakdown + validation-tuned threshold selection**, per
   model (causal / no-graph / correlation-graph), before trusting F1 further.
2. **Multi-seed stability check** (2-3 seeds) on the causal model's advantage.
3. **Backend API** (`/api/predict`, `/api/refresh_live`) wiring the real
   checkpoint to `stock_prediction_web.html`, exactly per constraints 1, 6, 7, 8, 9
   above.
4. Only after 1-3: update the paper/report with real, honestly-caveated numbers.
