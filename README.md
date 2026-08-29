# 🌋 Volatility MCP

> An MCP (Model Context Protocol) server for iterative volatility analytics: ARCH → GARCH → GJR-GARCH → EGARCH, with pre-flight statistical gates, input optimization, VaR / Expected Shortfall, Basel backtesting, and a stateful feedback loop that guides an LLM agent through every stage of the analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)

> **Volatility Analytics Lab** · Finance · Risk · Analytics

---

## What This Is

A **feedback-driven** volatility analytics engine wrapped as an MCP server. An LLM agent (Claude, GPT, etc.) calls tools one at a time; each tool returns results **plus recommendations for the next step**. The agent iterates until diagnostics pass, backtests pass, and reports are generated.

```
data.load → preflight.run → optimize.* → models.fit → diagnostics.run
    → compare.run → risk.var → backtest.rolling → backtest.coverage
    → report.excel → feedback.explain_decision
```

### Three Pillars

| Pillar | What it does | Folder |
|---|---|---|
| 🛡️ **Pre-flight gates** | 12 statistical checks run BEFORE any model fit. If Engle ARCH-LM fails, GARCH is blocked. | `core/preflight/` |
| 🎯 **Input optimization** | `(p,q)`, distribution, window, refit frequency chosen by AIC/QLIKE — not by guesswork. | `core/optimize/` |
| 🔁 **Feedback loop** | Session state + workflow DAG + advisor. Every tool returns `next_actions[]`. | `core/feedback/` |

---

## Quick Start

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

### Connect from Claude Desktop

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

## Tool Surface

| Category | Tools |
|---|---|
| **Session** | `open_session`, `close_session`, `get_session_state`, `list_sessions`, `reset_session` |
| **Data** | `load_market`, `load_csv`, `list_universes`, `list_assets` |
| **Pre-flight** | `run_preflight`, `get_gate_status`, `explain_gate` |
| **Optimize** | `optimize_order`, `optimize_distribution`, `optimize_mean`, `optimize_window`, `optimize_refit`, `optimize_horizon` |
| **Models** | `fit_model`, `forecast`, `list_models` |
| **Diagnostics** | `run_diagnostics`, `get_diagnostics` |
| **Compare** | `compare_models` |
| **Risk** | `compute_var`, `compute_es`, `basel_es`, `portfolio_var` |
| **Scenario** | `apply_market_shock`, `stressed_var`, `list_stress_scenarios` |
| **Backtest** | `rolling_backtest`, `backtest_kupiec`, `backtest_christoffersen`, `backtest_traffic_light`, `backtest_dynamic_quantile`, `backtest_diebold_mariano`, `backtest_coverage_scorecard` |
| **Report** | `build_excel`, `build_pdf`, `build_markdown` |
| **Feedback** | `get_next_action`, `explain_decision`, `get_recommendations`, `list_valid_transitions` |

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

## Folder Structure

```
volatility-mcp/
├── mcp_server/                   # MCP protocol layer
│   ├── server.py                 # Entry point: registers tools, resources, prompts
│   ├── tools/                    # Tools callable by the LLM agent
│   │   ├── session.py            # open_session / close_session / get_state
│   │   ├── data.py               # load_market / load_csv / list_universes
│   │   ├── preflight.py          # run_preflight / get_gate_status
│   │   ├── diagnostics.py        # run_diagnostics (LB, ARCH-LM, JB, Q-Q stats)
│   │   ├── optimize.py           # optimize_order / optimize_distribution / optimize_window
│   │   ├── models.py             # fit_model / forecast / list_models
│   │   ├── compare.py            # compare_models (AIC/BIC/QLIKE/RMSE)
│   │   ├── risk.py               # compute_var / compute_es / basel_es
│   │   ├── scenario.py           # apply_shock / stressed_var
│   │   ├── backtest.py           # rolling_backtest / kupiec / christoffersen
│   │   ├── report.py             # build_excel / build_pdf / build_markdown
│   │   └── feedback.py           # get_next_action / explain_decision
│   ├── resources/                # Static context (read by LLM, not executed)
│   │   ├── model_catalog.md
│   │   ├── test_catalog.md
│   │   └── workflows/            # 01_discovery → 05_validation
│   ├── prompts/                  # Prompt templates for guided workflows
│   └── schemas/                  # JSON schemas for tool I/O contracts
│
├── core/                         # Framework-agnostic engine (no MCP code here)
│   ├── data/                     # Providers: yfinance, CSV, Alpha Vantage, Polygon, Kite
│   ├── preflight/                # 12-check gate layer
│   │   └── checks/               # 01_sample_size.py … 12_frequency_adequacy.py
│   ├── diagnostics/              # Post-fit: LB, ARCH-LM, JB, sign-bias, Nyblom
│   ├── models/
│   │   ├── univariate/           # ARCH, GARCH, GJR-GARCH, EGARCH, FIGARCH, EWMA
│   │   ├── multivariate/         # DCC, BEKK, O-GARCH
│   │   ├── stochastic/           # Heston, SV-Jumps
│   │   └── ml/                   # LSTM, Transformer, TFT
│   ├── optimize/                 # order_selector, distribution_selector, window_selector…
│   ├── risk/                     # VaR (parametric, historical, FHS, MC), ES, Basel, portfolio
│   ├── backtest/                 # Kupiec, Christoffersen, DQ, Traffic Light, Diebold-Mariano
│   ├── feedback/                 # Session, state machine, workflow DAG, advisor, decision log
│   └── reporting/                # Excel, PDF, Markdown, Plotly charts
│
├── session_store/                # SQLite / DuckDB persistent session backend
├── tests/                        # Mirrors core/ structure; synthetic GARCH fixtures
├── examples/
│   ├── 01_basic_flow.py          # preflight → fit → forecast → VaR
│   ├── 02_feedback_loop_demo.py  # Full iterate-until-stable flow
│   ├── 03_optimization_walkthrough.py
│   └── notebooks/mcp_client_demo.ipynb
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
└── docs/
    ├── ARCHITECTURE.md
    ├── FEEDBACK_LOOP.md
    ├── PREFLIGHT_CHECKS.md
    ├── INPUT_OPTIMIZATION.md
    ├── MODEL_CATALOG.md
    ├── TEST_CATALOG.md
    ├── MCP_PROTOCOL.md
    └── DECISION_RULES.md
```

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
## Credits & Intellectual Debt

This project is built on the shoulders of the following thinkers and their work.

**Nassim Nicholas Taleb** — *The Black Swan* (2007), *Dynamic Hedging* (1997),
*Antifragile* (2012). The pre-flight normality gate, stressed VaR scenarios,
and the philosophy of iterating toward robustness over point estimates all
trace back here.

**Robert F. Engle** — ARCH paper (*Econometrica*, 1982). The Engle ARCH-LM
test is the single most critical pre-flight gate: if it fails, GARCH is blocked.

**Tim Bollerslev** — GARCH(*p,q*) paper (*Journal of Econometrics*, 1986).
The industry workhorse at the center of the model catalog.

**Nelson (1991), Glosten-Jagannathan-Runkle (1993)** — EGARCH and GJR-GARCH
respectively. The leverage asymmetry gate recommends these when sign-bias fires.

**Philippe Jorion** — *Value at Risk* (3rd ed., 2006). The parametric,
historical, and FHS VaR implementations follow his treatment directly.

**Kupiec (1995), Christoffersen (1998), Engle & Manganelli (2004)** — the
POF, conditional coverage, and Dynamic Quantile backtests that form the
coverage scorecard.

**Patton (2011)** — established QLIKE as the robust loss function for
volatility forecast comparison. Default objective in `optimize_window`.

**Diebold & Mariano (1995)** — the DM test used in `backtest_diebold_mariano`
to compare competing model forecasts.

**Basel Committee on Banking Supervision** — Basel III framework: source of
the 97.5% ES requirement, 10-day horizon, traffic light zones, and capital
multiplier rules.

**Kevin Sheppard** — the [`arch`](https://github.com/bashtage/arch) Python
package that powers all GARCH-family estimation in this project.

---

## License

MIT — see [LICENSE](LICENSE).

Educational use only. Market data may be delayed. Not investment advice.
