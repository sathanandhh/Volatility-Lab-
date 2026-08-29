# MCP Protocol Surface

> Everything the LLM agent sees: tools, resources, and prompts.

## Tools (18 categories, 40+ individual tools)

### Session
| Tool | Args | Returns |
|---|---|---|
| `open_session` | — | session_id, stage, next_actions |
| `close_session` | session_id | status |
| `get_session_state` | session_id | full session state + next_actions |
| `list_sessions` | — | list of active sessions |
| `reset_session` | session_id, keep_data | reset session state |

### Data
| Tool | Args | Returns |
|---|---|---|
| `load_market` | session_id, ticker, years, frequency | n_obs, date range, next_actions |
| `load_csv` | session_id, path, date_column, value_column, is_returns | n_obs, date range |
| `list_universes` | — | {universe: [instruments]} |
| `list_assets` | universe? | {name: ticker} |

### Pre-flight
| Tool | Args | Returns |
|---|---|---|
| `run_preflight` | session_id | GateResult (12 checks) |
| `get_gate_status` | session_id | last GateResult |
| `explain_gate` | session_id, check_name | plain-language explanation |

### Optimize
| Tool | Args | Returns |
|---|---|---|
| `optimize_order` | session_id, model_family, max_p, max_q, distribution, criterion | optimal (p,q) + sweep |
| `optimize_distribution` | session_id, model_family, p, q, candidates | optimal dist + sweep |
| `optimize_mean` | session_id, candidates | optimal mean spec |
| `optimize_window` | session_id, min, max, step, metric | optimal window |
| `optimize_refit` | session_id, model_name, min, max, alpha | optimal refit freq |
| `optimize_horizon` | session_id, target_use | recommended horizon |
| `get_optimization_summary` | session_id | all optimization results |

### Models
| Tool | Args | Returns |
|---|---|---|
| `list_models` | family? | list of model specs |
| `fit_model` | session_id, model, p?, q?, o?, distribution?, mean?, force? | fit result |
| `forecast` | session_id, model, horizon, method, simulations | variance path |

### Diagnostics
| Tool | Args | Returns |
|---|---|---|
| `run_diagnostics` | session_id, model_name | full diagnostic suite |
| `get_diagnostics` | session_id, model_name | cached diagnostics |

### Compare
| Tool | Args | Returns |
|---|---|---|
| `compare_models` | session_id | scorecard with ranks |

### Risk
| Tool | Args | Returns |
|---|---|---|
| `compute_var` | session_id, model, confidence, horizon, portfolio_value, method | VaR |
| `compute_es` | session_id, model, confidence, horizon, portfolio_value, method | ES + ES/VaR ratio |
| `basel_es` | session_id, model, portfolio_value, base_horizon | Basel 97.5% ES |
| `portfolio_var` | session_id, weights, confidence, portfolio_value, method | marginal/component/incremental |

### Scenario
| Tool | Args | Returns |
|---|---|---|
| `apply_market_shock` | session_id, model, shock_pct, direction | shocked vol forecast |
| `stressed_var` | session_id, model, confidence, portfolio_value, scenario_name | stressed VaR |
| `list_stress_scenarios` | — | list of crisis scenarios |

### Backtest
| Tool | Args | Returns |
|---|---|---|
| `rolling_backtest` | session_id, models, confidence, test_fraction, refit_every? | per-model breaches |
| `backtest_kupiec` | session_id, model | Kupiec stat + p |
| `backtest_christoffersen` | session_id, model | independence + CC |
| `backtest_traffic_light` | session_id, model | Basel zone |
| `backtest_dynamic_quantile` | session_id, model | DQ stat + p |
| `backtest_diebold_mariano` | session_id, model_a, model_b | DM stat + p |
| `backtest_coverage_scorecard` | session_id | combined scorecard |

### Report
| Tool | Args | Returns |
|---|---|---|
| `build_excel` | session_id, output_path? | file path |
| `build_pdf` | session_id, output_path? | file path |
| `build_markdown` | session_id | markdown content |

### Feedback
| Tool | Args | Returns |
|---|---|---|
| `get_next_action` | session_id, intent? | recommended action + args + alternatives |
| `explain_decision` | session_id, decision_id? | audit trail |
| `get_recommendations` | session_id | heuristic recommendations |
| `list_valid_transitions` | session_id | valid/invalid actions |

## Resources (read-only context)

| URI | Content |
|---|---|
| `volatility://methodology` | Master methodology document |
| `volatility://models/catalog` | Model catalog with specs |
| `volatility://tests/catalog` | Statistical test catalog |
| `volatility://distributions/catalog` | Distribution catalog |
| `volatility://workflows/01_discovery` | Discovery workflow |
| `volatility://workflows/02_modeling` | Modeling workflow |
| `volatility://workflows/03_optimization` | Optimization workflow |
| `volatility://workflows/04_risk` | Risk quantification workflow |
| `volatility://workflows/05_validation` | Validation & reporting workflow |

## Prompts

| Prompt | Args | Use case |
|---|---|---|
| `guided_volatility_analysis` | asset, years | Full guided workflow |
| `troubleshoot_fit_failure` | model, error | Debug a failed fit |
| `explain_results_to_student` | audience | Plain-language explanation |
