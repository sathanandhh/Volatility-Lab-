# Workflow 1 · Discovery

**Goal:** understand what volatility looks like in real markets before
fitting any model.

## Steps

1. `session.open_session` → obtain `session_id`
2. `data.list_universes` → pick a market group
3. `data.load_market` → load 5+ years of returns
4. `preflight.run` → see the 12 stylized facts as statistical tests
5. Read pre-flight results carefully:
   - arch_effect pass? → GARCH is justified
   - leverage_asymmetry pass? → GJR/EGARCH recommended
   - normality block? → Student-t distribution mandatory
6. `feedback.get_next_action` → decide whether to optimize or fit

## What to look for

- **Time variation** in rolling volatility (visible in conditional_volatility tail)
- **Clustering** → arch_effect + volatility_clustering tests
- **Persistence** → ACF decay (visible in arch_effect p-value strength)
- **Mean reversion** → long-run anchor vs rolling
- **Fat tails** → normality tests + excess kurtosis
- **Asymmetry** → sign-bias + leverage tests

## Exit criterion

Preflight overall status is `pass` or `warn` (not `block`). If `block`,
follow the recommendation to fix the data or use a different model
family (e.g., EWMA if no ARCH effect).
