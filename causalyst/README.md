# Causalyst — working code scaffold

# Causalyst — working code scaffold

## What's REAL and already run (not synthetic, not fabricated)

- `src/real_data_experiment.py` + `data/nse_real/*.csv` — actual daily OHLCV data for 10 real NSE-listed stocks (RELIANCE, TCS, INFY, HCLTECH, WIPRO, HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, SBIN), 2012–2021, sourced from a public GitHub mirror of NSE data (Ratnesh-bhosale/NIFTY500_dataset). Results saved in `data/real_data_results.json`.
- `src/pcmci_discovery.py` — real PCMCI (multivariate causal discovery, via `tigramite`) run on the same real data. Results in `data/pcmci_results.json`.
- Both are genuinely run and reported in the paper's Section 5.2, with an explicit scope note: this validates the **causal-discovery methodology on price data only** — it does NOT include institutional flow (FII/DII), does NOT include a prediction model, and is NOT a backtest.

Rerun them yourself:
```bash
cd src
python3 real_data_experiment.py   # correlation vs Granger vs PCMCI on real stocks, + stability check
python3 pcmci_discovery.py        # PCMCI in isolation
```

## What's real and runs on synthetic data (ground-truth validated)

- `src/synthetic_data.py` — synthetic data with a KNOWN causal structure
- `src/causal_discovery.py` — Granger causality (Bonferroni-corrected) + correlation baseline, validated against that ground truth

## What needs your API keys / accounts

- `src/agents.py` — multi-agent debate. Needs `ANTHROPIC_API_KEY`. Structure and prompts are complete; not yet run.

## What is genuinely NOT done — do not present these as finished

1. **FII/DII institutional-flow data** — could NOT be freely obtained in bulk historical form. See `src/fii_dii_scraper_TEMPLATE.py` — an UNTESTED starter script (our sandbox can't reach nseindia.com) for forward data collection. This means the paper's central "institutional flow as causal node" claim is validated on synthetic data only. Say this plainly in the paper — don't imply otherwise.
2. **Temporal GNN predictor** — not implemented. No prediction accuracy numbers exist anywhere in this repo.
3. **Walk-forward backtest** — not run, because there's no predictor yet to backtest.
4. **Debate ablation** (single-agent vs. naive debate vs. our heterogeneous design) — not run. This is the experiment that would justify Section 4.5's architecture; right now it's a design, not a validated result.
5. **Guide review, venue selection, plagiarism check, submission** — all still need you and your guide.

See the paper's Appendix A for the same list, so it's visible to anyone reading the draft.

