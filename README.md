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

## License

MIT — see [LICENSE](LICENSE).

Educational use only. Market data may be delayed.Not investment advice.
