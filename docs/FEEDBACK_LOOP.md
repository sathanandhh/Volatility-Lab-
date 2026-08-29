# The Feedback Loop

> How the iterative workflow works in the Volatility MCP.

## Why a feedback loop?

Traditional analytics tools are **single-shot**: the user provides
inputs, the tool produces outputs, and if the outputs are wrong the
user must re-run with different inputs. This is fragile because:

- Financial data has **stylized facts** (clustering, fat tails, leverage)
  that require specific model choices.
- The **right model** depends on the data — you can't know in advance
  whether GARCH or EGARCH is better.
- **Backtests fail** when the model is mis-calibrated, and the fix
  (shorter refit, heavier tail, different order) is data-dependent.

The MCP server solves this by making every tool call return
`next_actions[]` and `recommendations[]`. The agent reads these and
decides the next step. If diagnostics fail, the loop returns to
optimize. If backtests fail, the loop returns to re-tune. This
continues until all tests pass or the agent decides to stop.

## The workflow DAG

```
        empty ──data──▶ data ──preflight──▶ preflight
                                              │
                                   (block? → explain_gate)
                                              │
                                              ▼
                                          optimize ──fit──▶ fit
                                              ▲              │
                                              │              ▼
                                              └──────────compare
                                                              │
                                                              ▼
                                                            risk
                                                              │
                                                              ▼
                                                          backtest
                                                              │
                                                ┌─────────────┤
                                                ▼             ▼
                                          (fail → re-tune)  report
                                                │             │
                                                └──▶ optimize │
                                                    (loop)    │
                                                              ▼
                                                        explain_decision
```

## Session state

Every tool call takes a `session_id`. The session stores:

| Attribute | Type | Purpose |
|---|---|---|
| `returns` | pd.Series | The loaded return series |
| `preflight_result` | GateResult | 12-check aggregated result |
| `optimization_results` | dict[str, OptimizationResult] | Per-input optimum |
| `fitted_models` | dict[str, FitResult] | All fitted models |
| `diagnostics_results` | dict[str, dict] | Per-model diagnostics |
| `backtest_results` | dict[str, ModelBacktestResult] | Walk-forward results |
| `risk_results` | dict[str, dict] | VaR / ES / Basel results |
| `coverage_scorecard` | list[dict] | Combined backtest verdict |
| `decision_log` | DecisionLog | Append-only audit trail |
| `current_stage` | str | Where in the DAG we are |

## The advisor

`feedback.get_next_action(session_id)` inspects the session and returns:

```json
{
  "recommended_action": "models.fit",
  "suggested_args": {"model": "EGARCH", "p": 1, "q": 1, "dist": "t"},
  "rationale": "Preflight passed and order/distribution optimized.",
  "alternatives": [
    {"action": "models.fit", "args": {"model": "GARCH"},
     "why": "Try symmetric GARCH first for comparison"}
  ],
  "blocked_actions": [
    {"action": "diagnostics.run", "reason": "No fitted model yet"}
  ],
  "current_stage": "optimize",
  "decision_count": 5
}
```

## Heuristic recommendations

The advisor uses both **state-machine logic** (which stage are we in?)
and **heuristic rules** (what do the test results say?):

| Condition | Recommendation |
|---|---|
| Preflight `arch_effect` = block | Use EWMA; do NOT fit GARCH |
| Preflight `normality` = block | Use Student-t distribution |
| Preflight `leverage_asymmetry` = pass | Try GJR-GARCH or EGARCH |
| Diagnostics: LB(r²) p < 0.05 | Increase q; try FIGARCH |
| Diagnostics: ARCH-LM p < 0.05 | Model misspecified |
| Diagnostics: JB p < 0.01 | Switch distribution |
| Diagnostics: sign-bias p < 0.05 | Try GJR/EGARCH |
| Backtest: Kupiec p < 0.05 | Re-tune distribution or window |
| Backtest: Christoffersen p < 0.05 | Shorten refit interval |
| Backtest: Traffic light = red | Capital multiplier 4.0× |

## Example loop

```
Agent:   load_market(^NSEI, 5y)
MCP:     ✓ 1234 obs loaded. next_actions=["preflight.run"]

Agent:   run_preflight()
MCP:     ⚠ overall=warn (normality blocked, leverage detected)
         recommendations: ["Use Student-t", "Try GJR-GARCH or EGARCH"]
         next_actions=["optimize.distribution", "optimize.order"]

Agent:   optimize_distribution()
MCP:     ✓ Best: Student-t (AIC=3412 vs Normal AIC=3445)

Agent:   optimize_order()
MCP:     ✓ Best: p=1, q=1 (AIC=3412)

Agent:   fit_model(GARCH, p=1, q=1, dist=t)
MCP:     ✓ Converged. next_actions=["diagnostics.run", "compare.run"]

Agent:   run_diagnostics(model="GARCH")
MCP:     ⚠ ARCH-LM p=0.03 (remaining ARCH at lag 10)
         recommendation: "Increase q or try EGARCH"
         next_actions=["optimize.order", "models.fit"]

Agent:   fit_model(EGARCH, p=1, q=1, dist=t)   ← feedback loop iterates
MCP:     ✓ Converged. ARCH-LM p=0.21 (clean).

Agent:   compute_var(model="EGARCH", method="student_t")
MCP:     ✓ VaR = ₹234,000

Agent:   rolling_backtest(models=["EGARCH"])
         → coverage_scorecard
MCP:     ✓ Kupiec p=0.42, CC p=0.31, zone=green
         next_actions=["report.excel"]

Agent:   build_excel()
MCP:     ✓ Report saved to /tmp/volmcp_reports/volatility_report.xlsx

Agent:   explain_decision()
MCP:     Full audit trail of 8 decisions...
```

The loop is **not** infinite — the agent can stop at any point, and
the advisor will never recommend the same action twice without a
config change.
