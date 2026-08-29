# Workflow 5 · Validation & Reporting

**Goal:** statistically defend the VaR forecast and package the analysis.

## Prerequisites
- Workflow 4 complete (VaR computed)

## Steps

1. `optimize.optimize_window` — if not yet done
2. `optimize.optimize_refit` — if not yet done
3. `backtest.rolling_backtest` — walk-forward refit on test sample
4. `backtest.backtest_kupiec` — POF test per model
5. `backtest.backtest_christoffersen` — independence + CC
6. `backtest.backtest_traffic_light` — Basel TLB zone
7. `backtest.backtest_dynamic_quantile` — Engle-Manganelli DQ
8. `backtest.backtest_diebold_mariano` — pairwise model comparison
9. `backtest.backtest_coverage_scorecard` — combined verdict
10. `report.build_excel` — full workbook
11. `report.build_pdf` — regulator-ready PDF
12. `report.build_markdown` — LLM-consumable summary
13. `feedback.explain_decision` — full audit trail

## Pass criteria (Basel-style)

| Test | Pass |
|---|---|
| Kupiec POF p ≥ 0.05 | Required |
| Christoffersen independence p ≥ 0.05 | Required |
| Conditional coverage p ≥ 0.05 | Required |
| Traffic light zone | Green or Yellow |
| DQ p ≥ 0.05 | Recommended |

## Failure response (feedback loop)

| Failed test | Recommended action |
|---|---|
| Kupiec p < 0.05 (underprediction) | `optimize.optimize_window` (shorten) or `optimize.optimize_distribution` (heavier tail) |
| Christoffersen p < 0.05 (clustering) | `optimize.optimize_refit` (shorten interval) |
| DQ p < 0.05 | Try FHS or Monte Carlo VaR |
| NYblom unstable | Regime split; use rolling refit |

Iterate until pass, then package reports.
