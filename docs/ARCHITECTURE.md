# Architecture

> See also: [FEEDBACK_LOOP.md](FEEDBACK_LOOP.md), [PREFLIGHT_CHECKS.md](PREFLIGHT_CHECKS.md)

## Overview

The Volatility MCP is a **layered** system. Each layer has a single
responsibility and depends only on the layer below it.

```
┌─────────────────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                                 │
│  Claude Desktop · Streamlit · REST clients · Jupyter notebooks     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ MCP protocol (stdio / SSE / HTTP)
┌────────────────────────────────▼────────────────────────────────────┐
│  PROTOCOL LAYER                                                     │
│  FastMCP server                                                     │
│  • 18+ tools (callable by LLM)                                     │
│  • 5+ resources (read-only context)                                │
│  • 3 prompts (workflow templates)                                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ Python function calls
┌────────────────────────────────▼────────────────────────────────────┐
│  ENGINE LAYER (core/)                                              │
│  • preflight/  (12 statistical gates)                             │
│  • diagnostics/ (residual tests)                                   │
│  • models/     (ARCH → EGARCH, EWMA, multivariate, ML)            │
│  • optimize/   (order, distribution, window, refit selectors)      │
│  • risk/       (VaR, ES, Basel, stressed, scenario)               │
│  • backtest/   (Kupiec, Christoffersen, DQ, traffic light)       │
│  • feedback/   (session, state machine, advisor, DAG)            │
│  • reporting/  (Excel, PDF, Markdown)                             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ SQL + parquet
┌────────────────────────────────▼────────────────────────────────────┐
│  STORAGE LAYER                                                      │
│  • session_store/ (SQLite + parquet blobs)                        │
│  • In-memory cache (default)                                       │
│  • Redis (optional, for multi-worker)                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Design principles

1. **No model runs before pre-flight gates pass.** The Engle ARCH-LM
   test is the most important check — if it fails, the entire GARCH
   family is statistically unjustified and `models.fit` blocks.

2. **Inputs are optimized, not guessed.** `(p,q)`, distribution, window
   size, and refit frequency are chosen by information criteria (AIC,
   QLIKE, conditional-coverage p-value).

3. **Every tool returns `next_actions[]`.** The agent reads this list
   and decides the next call. `feedback.get_next_action` provides an
   explicit recommendation with suggested args.

4. **The feedback loop iterates.** If diagnostics fail → return to
   `optimize.*` or `models.fit`. If backtests fail → return to
   `optimize.refit` or `optimize.window`. The loop continues until all
   tests pass or the agent decides to stop.

5. **Core is framework-agnostic.** The `core/` package contains zero
   MCP-protocol code. It can be used from a CLI, a Jupyter notebook, a
   REST API, or the MCP server — all without modification.

## Data flow

```
User: "Analyze Reliance volatility"
    │
    ▼
Agent calls: open_session → load_market(RELIANCE.NS, 5y)
    │                    │
    │                    ▼
    │              core/data/service.py → yfinance → 1234 returns
    │
    ▼
Agent calls: run_preflight
    │              │
    │              ▼
    │        core/preflight/orchestrator.py
    │        runs 12 checks → GateResult{overall: pass, ...}
    │
    ▼
Agent calls: get_next_action → advisor says "optimize.order"
    │
    ▼
Agent calls: optimize_order → AIC sweep → optimal (p=1, q=1)
    │
    ▼
Agent calls: fit_model(GARCH, p=1, q=1, dist=t)
    │              │
    │              ▼
    │        core/models/univariate/arch_family.py → arch.arch_model.fit()
    │        → FitResult{params, conditional_volatility, std_resid, AIC, ...}
    │
    ▼
Agent calls: run_diagnostics
    │              │
    │              ▼
    │        core/diagnostics/ → LB, ARCH-LM, JB, sign-bias, Nyblom
    │        → if ARCH-LM p < 0.05: recommend re-optimize
    │
    ▼ (feedback loop — if diagnostics fail, go back to optimize/fit)
    │
Agent calls: compute_var → core/risk/var/parametric.py → VaR = ₹234,000
    │
    ▼
Agent calls: rolling_backtest → coverage_scorecard
    │              │
    │              ▼
    │        if Kupiec p < 0.05: recommend optimize.refit → loop back
    │
    ▼ (feedback loop — if backtest fails, re-tune refit)
    │
Agent calls: build_excel + build_markdown → reports
    │
    ▼
Agent calls: explain_decision → full audit trail → user
```

## Scaling options

| Scale | Backend | Transport | Session store |
|---|---|---|---|
| Single user (Claude Desktop) | yfinance | stdio | In-memory |
| Small team | yfinance + CSV | SSE | SQLite |
| Multi-worker | yfinance + Alpha Vantage | SSE + Nginx | SQLite + Redis cache |
| Enterprise | Bloomberg + Polygon | streamable-http + K8s | DuckDB + S3 |
