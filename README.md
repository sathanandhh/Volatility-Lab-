# Volatility MCP

An MCP (Model Context Protocol) server for iterative volatility analytics: ARCH → GARCH → GJR-GARCH → EGARCH, with pre-flight statistical gates, input optimization, VaR / Expected Shortfall, Basel backtesting, and a stateful feedback loop that guides an LLM agent through every stage of the analysis.

> **Volatility Analytics Lab** · Finance · Risk · Analytics

---

## What this is

A **feedback-driven** volatility analytics engine wrapped as an MCP server. An LLM agent (Claude, GPT, etc.) calls tools one at a time; each tool returns results **plus recommendations for the next step**. The agent iterates until diagnostics pass, backtests pass, and reports are generated.

```
data.load → preflight.run → optimize.* → models.fit → diagnostics.run
    → compare.run → risk.var → backtest.rolling → backtest.coverage
    → report.excel → feedback.explain_decision
```

### Three pillars

| Pillar | What it does | Folder |
|---|---|---|
| **Pre-flight gates** | 12 statistical checks run BEFORE any model fit. If Engle ARCH-LM fails, GARCH is blocked. | `core/preflight/` |
| **Input optimization** | `(p,q)`, distribution, window, refit frequency chosen by AIC/QLIKE — not by guesswork. | `core/optimize/` |
| **Feedback loop** | Session state + workflow DAG + advisor. Every tool returns `next_actions[]`. | `core/feedback/` |

---

## Quick start

```bash
# Clone
git clone https://github.com/volatility-analytics-lab/volatility-mcp.git
cd volatility-mcp

# Install
python -m pip install -e ".[dev]"

# Run tests
make test

# Start the MCP server (stdio transport)
make run

# Or SSE transport for remote access
make run-sse
```

---

## Connect from Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "volatility": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/volatility-mcp"
    }
  }
}
```

Then ask Claude: *"Analyze Reliance Industries volatility using the volatility MCP server."*

---

## Tool surface (18 tools)

| Category | Tools |
|---|---|
| Session | `open_session`, `close_session`, `get_session_state`, `list_sessions`, `reset_session` |
| Data | `load_market`, `load_csv`, `list_universes`, `list_assets` |
| Pre-flight | `run_preflight`, `get_gate_status`, `explain_gate` |
| Optimize | `optimize_order`, `optimize_distribution`, `optimize_mean`, `optimize_window`, `optimize_refit`, `optimize_horizon` |
| Models | `fit_model`, `forecast`, `list_models` |
| Diagnostics | `run_diagnostics`, `get_diagnostics` |
| Compare | `compare_models` |
| Risk | `compute_var`, `compute_es`, `basel_es`, `portfolio_var` |
| Scenario | `apply_market_shock`, `stressed_var`, `list_stress_scenarios` |
| Backtest | `rolling_backtest`, `backtest_kupiec`, `backtest_christoffersen`, `backtest_traffic_light`, `backtest_dynamic_quantile`, `backtest_diebold_mariano`, `backtest_coverage_scorecard` |
| Report | `build_excel`, `build_pdf`, `build_markdown` |
| Feedback | `get_next_action`, `explain_decision`, `get_recommendations`, `list_valid_transitions` |

See [docs/MCP_PROTOCOL.md](docs/MCP_PROTOCOL.md) for the full surface.

---

## Architecture

```
Presentation:  Streamlit · Claude Desktop · REST clients · Jupyter
                              │ HTTP / stdio / SSE
Protocol:       FastMCP server (tools · resources · prompts)
                              │
Engine:         core/  (preflight · diagnostics · models · optimize
                        · risk · backtest · feedback · reporting)
                              │
Storage:        session_store/ (SQLite + parquet)
Data:           yfinance · CSV · Alpha Vantage · Polygon · Kite
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full layered design.

---

## Documentation

| Doc | What it covers |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layered architecture, data flow, scaling options |
| [FEEDBACK_LOOP.md](docs/FEEDBACK_LOOP.md) | How the iterative workflow works |
| [PREFLIGHT_CHECKS.md](docs/PREFLIGHT_CHECKS.md) | Every gate explained |
| [INPUT_OPTIMIZATION.md](docs/INPUT_OPTIMIZATION.md) | How each input is tuned |
| [MODEL_CATALOG.md](docs/MODEL_CATALOG.md) | When to use which model |
| [TEST_CATALOG.md](docs/TEST_CATALOG.md) | Every statistical test |
| [MCP_PROTOCOL.md](docs/MCP_PROTOCOL.md) | Tool/resource/prompt surface |
| [DECISION_RULES.md](docs/DECISION_RULES.md) | Decision tree for model selection |

---

## Folder Structure

volatility-mcp/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── Makefile
├── LICENSE
│
├── mcp_server/                          # ── MCP protocol layer ──────────────
│   ├── __init__.py
│   ├── server.py                         # Entry point: registers tools, resources, prompts
│   │
│   ├── tools/                            # Tools callable by the LLM agent
│   │   ├── __init__.py
│   │   ├── session.py                   # open_session / close_session / get_state
│   │   ├── data.py                      # load_market / load_csv / list_universes
│   │   ├── preflight.py                 # run_preflight / get_gate_status
│   │   ├── diagnostics.py               # run_diagnostics (LB, ARCH-LM, JB, Q-Q stats)
│   │   ├── optimize.py                  # optimize_order / optimize_distribution
│   │   │                                # optimize_window / optimize_refit
│   │   ├── models.py                    # fit_model / forecast / list_models
│   │   ├── compare.py                   # compare_models (AIC/BIC/QLIKE/RMSE)
│   │   ├── risk.py                      # compute_var / compute_es / basel_es
│   │   ├── scenario.py                  # apply_shock / stressed_var
│   │   ├── backtest.py                  # rolling_backtest / kupiec / christoffersen
│   │   ├── report.py                    # build_excel / build_pdf / build_markdown
│   │   └── feedback.py                  # get_next_action / explain_decision
│   │
│   ├── resources/                       # Static context (read by LLM, not executed)
│   │   ├── methodology.md
│   │   ├── model_catalog.md             # ARCH, GARCH, GJR, EGARCH, FIGARCH, DCC…
│   │   ├── test_catalog.md              # Every statistical test: when, why, how
│   │   ├── distribution_catalog.md      # Normal, t, GED, skew-t, GHD
│   │   └── workflows/
│   │       ├── 01_discovery.md
│   │       ├── 02_modeling.md
│   │       ├── 03_optimization.md
│   │       ├── 04_risk.md
│   │       └── 05_validation.md
│   │
│   ├── prompts/                         # Prompt templates for guided workflows
│   │   ├── guided_volatility_analysis.md
│   │   ├── troubleshoot_fit_failure.md
│   │   └── explain_results_to_student.md
│   │
│   └── schemas/                         # JSON schemas for tool I/O contracts
│       ├── preflight_result.json
│       ├── diagnostics_result.json
│       ├── optimization_result.json
│       ├── fit_result.json
│       ├── backtest_result.json
│       └── feedback_result.json
│
├── core/                                # ── Framework-agnostic engine ───────
│   ├── __init__.py
│   ├── config.py                        # Pydantic settings
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── service.py                   # Unified DataProvider interface
│   │   ├── providers/
│   │   │   ├── base.py
│   │   │   ├── yfinance_provider.py
│   │   │   ├── csv_provider.py
│   │   │   ├── alpha_vantage_provider.py
│   │   │   ├── polygon_provider.py
│   │   │   └── kite_provider.py         # Indian markets
│   │   ├── transforms.py                # Log returns, resampling, scaling
│   │   └── quality.py                   # Gap detection, duplicates, dtype sanity
│   │
│   ├── preflight/                       # ── PRE-FLIGHT GATE LAYER ──────────
│   │   ├── __init__.py
│   │   ├── orchestrator.py              # Runs all checks, aggregates gate result
│   │   ├── gates.py                      # Pass / Warn / Block decision logic
│   │   │
│   │   └── checks/
│   │       ├── __init__.py
│   │       ├── 01_sample_size.py        # Min 300 (GARCH) / 500 (EGARCH)
│   │       ├── 02_missing_data.py       # Max gap, % missing, imputation need
│   │       ├── 03_outliers.py           # Hampel, rolling 4σ, MAD
│   │       ├── 04_zero_infinity.py       # Bad values (WTI April-2020 case)
│   │       ├── 05_stationarity.py       # ADF + KPSS on returns
│   │       ├── 06_structural_break.py    # Bai-Perron, CUSUM, Chow
│   │       ├── 07_arch_effect.py         # Engle LM — IS GARCH EVEN NEEDED?
│   │       ├── 08_volatility_clustering.py  # LB on |r| and r²
│   │       ├── 09_normality.py          # JB, Shapiro-Wilk, Anderson-Darling
│   │       ├── 10_mean_specification.py  # LB on residuals → AR/ARMA need?
│   │       ├── 11_leverage_asymmetry.py  # Sign-bias test → GJR/EGARCH need?
│   │       └── 12_frequency_adequacy.py  # Too few obs for weekly/monthly?
│   │
│   ├── diagnostics/                     # Post-fit residual diagnostics
│   │   ├── __init__.py
│   │   ├── residual.py                  # Std residuals, LB, LB²
│   │   ├── arch_lm.py                   # Remaining ARCH effect
│   │   ├── information_criteria.py      # AIC, BIC, HQIC
│   │   ├── accuracy.py                  # QLIKE, RMSE, MAE, MAFE
│   │   ├── normality.py                 # JB, KS, A-D on std residuals
│   │   ├── sign_bias.py                 # Negative/positive sign bias
│   │   └── nyblom.py                    # Parameter stability
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py                  # ModelSpec dataclass, catalog
│   │   ├── univariate/
│   │   │   ├── arch_family.py           # ARCH, GARCH, GJR-GARCH, EGARCH
│   │   │   ├── figarch.py               # Fractionally integrated
│   │   │   ├── igarch.py
│   │   │   ├── ewma.py
│   │   │   ├── realized_garch.py
│   │   │   └── heavy.py
│   │   ├── multivariate/
│   │   │   ├── dcc.py
│   │   │   ├── bekk.py
│   │   │   └── ogarch.py
│   │   ├── stochastic/
│   │   │   ├── heston.py
│   │   │   └── sv_jumps.py
│   │   ├── ml/
│   │   │   ├── lstm_vol.py
│   │   │   ├── transformer_vol.py
│   │   │   └── tft.py                   # Temporal Fusion Transformer
│   │   └── distributions.py             # Normal, t, GED, skew-t, GHD, JSU
│   │
│   ├── optimize/                        # ── INPUT OPTIMIZATION LAYER ──────
│   │   ├── __init__.py
│   │   ├── optimizer.py                 # Base Optimizer interface
│   │   ├── grid_search.py
│   │   ├── bayesian_opt.py              # Optuna integration
│   │   ├── order_selector.py           # p,q via AIC/BIC sweep
│   │   ├── distribution_selector.py    # Normal/t/GED/skew-t via AIC
│   │   ├── mean_selector.py            # Constant/AR/ARMA via residual LB
│   │   ├── window_selector.py          # Rolling window via QLIKE
│   │   ├── refit_selector.py           # Refit frequency via coverage test
│   │   ├── horizon_selector.py         # Forecast horizon guidance
│   │   └── heuristics.py               # Rules-of-thumb docstring bank
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── var/
│   │   │   ├── parametric.py            # Normal, t, CF
│   │   │   ├── historical.py
│   │   │   ├── filtered_hist.py        # FHS
│   │   │   └── monte_carlo.py
│   │   ├── es/
│   │   │   ├── expected_shortfall.py
│   │   │   └── basel_es.py             # 97.5% ES, liquidity scaling
│   │   ├── stressed.py                 # Stressed VaR/ES,scenario library
│   │   ├── scenario.py                  # What-if shocks
│   │   ├── portfolio.py                # Marginal/component/incremental VaR
│   │   └── attribution.py              # Risk decomposition
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── rolling.py                   # Walk-forward engine
│   │   ├── kupiec.py                    # POF test
│   │   ├── christoffersen.py            # Independence + CC
│   │   ├── traffic_light.py            # Basel TLB
│   │   ├── dynamic_quantile.py          # Engle-Manganelli DQ
│   │   ├── diebold_mariano.py          # Forecast comparison
│   │   └── coverage.py                 # Combined coverage scorecard
│   │
│   ├── feedback/                        # ── FEEDBACK LOOP ENGINE ──────────
│   │   ├── __init__.py
│   │   ├── session.py                   # Session/workspace object (stateful)
│   │   ├── state.py                     # State machine: transitions between actions
│   │   ├── workflow_dag.py              # DAG of valid action transitions
│   │   ├── advisor.py                   # LLM-facing next-step advisor
│   │   ├── recommendations.py           # Heuristic recommendation engine
│   │   ├── decision_log.py              # Append-only log of decisions
│   │   └── defaults.py                  # Default next-action mappings
│   │
│   └── reporting/
│       ├── __init__.py
│       ├── excel.py                     # xlsxwriter workbooks
│       ├── pdf.py                       # WeasyPrint / ReportLab
│       ├── markdown.py
│       └── charts.py                   # Plotly → static PNG for PDF
│
├── session_store/                       # Persistent session storage
│   ├── __init__.py
│   ├── store.py                         # DuckDB / SQLite backend
│   ├── migrations/
│   │   └── 001_init.sql
│   └── schemas.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Fixtures: synthetic GARCH data
│   ├── test_preflight/
│   │   ├── test_sample_size.py
│   │   ├── test_arch_effect.py
│   │   ├── test_structural_break.py
│   │   └── test_gate_logic.py
│   ├── test_diagnostics/
│   ├── test_models/
│   ├── test_optimize/
│   ├── test_risk/
│   ├── test_backtest/
│   ├── test_feedback/
│   │   ├── test_state_transitions.py
│   │   └── test_advisor.py
│   └── test_mcp/
│       └── test_tools.py
│
├── examples/
│   ├── 01_basic_flow.py                 # preflight → fit → forecast → VaR
│   ├── 02_feedback_loop_demo.py         # Full iterate-until-stable flow
│   ├── 03_optimization_walkthrough.py
│   └── notebooks/
│       └── mcp_client_demo.ipynb
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
│
└── docs/
    ├── ARCHITECTURE.md
    ├── FEEDBACK_LOOP.md                  # How the iterate loop works
    ├── PREFLIGHT_CHECKS.md              # Every gate explained
    ├── INPUT_OPTIMIZATION.md            # How each input is tuned
    ├── MODEL_CATALOG.md
    ├── TEST_CATALOG.md
    ├── MCP_PROTOCOL.md                   # Tool/resource/prompt surface
    └── DECISION_RULES.md                 # When to use which model
---    

## License

MIT — see [LICENSE](LICENSE).

Educational use only. Market data may be delayed.Not investment advice.
