A model fit failed. Diagnose the cause and propose a fix.

## Failure details

- Model: **{model}**
- Error: ```{error}```

## Diagnostic checklist

Walk through each common cause in order. For each, propose a concrete
fix the user can apply via the MCP server.

### 1. Sample size too small
- Cause: Fewer than 300 (GARCH) or 500 (EGARCH) observations.
- Check: `session.get_session_state` → `metadata.years` and `n_obs`.
- Fix: Increase `years` in `data.load_market`, or switch to a higher-
  frequency (Daily instead of Weekly).

### 2. Preflight blocks not addressed
- Cause: A preflight check returned `block` and `force=false` was used.
- Check: `preflight.get_gate_status`.
- Fix: Address each block (e.g., use Student-t distribution if
  normality is blocked) OR call `models.fit` with `force=true` after
  explaining the risk.

### 3. Non-stationarity
- Cause: ADF failed to reject unit root.
- Check: `preflight.explain_gate` with `check_name="stationarity"`.
- Fix: Difference the series, or use a mean specification with AR terms.

### 4. Bad values (zero, infinity, NaN)
- Cause: WTI April-2020 case, or stale data feed.
- Check: `preflight.explain_gate` with `check_name="zero_infinity"`.
- Fix: Filter non-positive prices in `data.load_csv` or pick a different
  instrument.

### 5. Distribution mismatch
- Cause: Fitting `normal` distribution to heavy-tailed data.
- Fix: Call `optimize.optimize_distribution` and use the winner.

### 6. Starting values / convergence
- Cause: Optimizer did not converge (convergence_flag ≠ 0).
- Fix: Try a different model family (e.g., GARCH instead of EGARCH),
  or increase `maxiter` in the fit (call `models.fit` with explicit args).

### 7. Multicollinearity in orders
- Cause: p and q both high → near-singular Hessian.
- Fix: Reduce `max_p` and `max_q` in `optimize.optimize_order`.

### 8. Structural break
- Cause: Bai-Perron detected a regime split.
- Fix: Split the sample at the break date and fit separately, or use a
  rolling refit (call `optimize.optimize_refit`).

## After fixing

Re-run `models.fit` and then `diagnostics.run`. If still failing, ask
the user to share the full session state via `session.get_session_state`.
