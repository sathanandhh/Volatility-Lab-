# Decision Rules

> When to use which model — a decision tree.

## Decision tree

```
1. Is there an ARCH effect? (Engle LM p < 0.10)
   │
   ├─ NO → Use EWMA or constant volatility. Do NOT fit GARCH.
   │
   └─ YES → Continue to step 2.
       │
2. Is there leverage / asymmetry? (sign-bias p < 0.05)
   │
   ├─ NO → Use symmetric GARCH(1,1). → step 3a
   │
   └─ YES → Continue to step 2b.
       │
       ├─ Are residuals heavy-tailed? (JB p < 0.001)
       │   │
       │   ├─ YES → EGARCH with Student-t. (best for leverage + tails)
       │   │
       │   └─ NO  → GJR-GARCH with Student-t.
       │
       └─ (if both pass, EGARCH is preferred for its positivity guarantee)

3a. Are residuals heavy-tailed? (JB p < 0.001)
    │
    ├─ YES → GARCH(1,1) with Student-t.
    │
    └─ NO  → GARCH(1,1) with Normal.

3b. Does the squared-return ACF decay very slowly? (long memory)
    │
    ├─ YES → FIGARCH.
    │
    └─ NO  → GARCH(1,1).

4. Is the series multivariate (≥ 2 assets)?
   │
   └─ YES → DCC-GARCH (for time-varying correlations).
       │
       └─ NO → stay with univariate.

5. Is intraday realized variance available?
   │
   └─ YES → Realized GARCH or HEAVY (uses RV as a regressor).
       │
       └─ NO → stay with daily GARCH.

6. Do you need closed-form option pricing?
   │
   └─ YES → Heston or Heston-Nandi.
       │
       └─ NO → stay with GARCH family.
```

## Quick reference table

| Situation | Recommended model | Recommended distribution |
|---|---|---|
| No ARCH effect | EWMA (λ=0.94) | Normal |
| ARCH effect, no leverage, normal resid | GARCH(1,1) | Normal |
| ARCH effect, no leverage, heavy tails | GARCH(1,1) | Student-t |
| ARCH effect, leverage, heavy tails | EGARCH | Student-t |
| ARCH effect, leverage, skew | GJR-GARCH | Skew-t |
| Long memory in vol | FIGARCH | Student-t |
| Multivariate | DCC-GARCH | per-asset |
| Intraday data available | Realized GARCH | — |
| Option pricing | Heston / Heston-Nandi | — |

## Distribution selection rules

| Residual property | Distribution |
|---|---|
| Passes Jarque-Bera | Normal |
| Fails JB, symmetric | Student-t |
| Fails JB, skewed | Skew-t |
| Very heavy tails | JSU or GHD |

## VaR method selection rules

| Situation | VaR method |
|---|---|
| Normal residuals, short horizon | Normal parametric |
| Student-t residuals | Student-t parametric |
| Skewed residuals | Cornish-Fisher |
| Non-parametric preference | Historical simulation |
| Want tail realism + conditional vol | FHS (filtered historical) |
| Path-dependent scenarios | Monte Carlo |
| Regulatory reference | Basel 97.5% ES |

## Backtest failure response

| Failed test | Fix |
|---|---|
| Kupiec p < 0.05 (wrong breach rate) | Re-tune distribution (heavier tail) or window (shorter) |
| Christoffersen p < 0.05 (clustering) | Shorten refit interval |
| DQ p < 0.05 | Switch to FHS or Monte Carlo VaR |
| Traffic light = red | Capital multiplier 4.0×; fundamental model review |
| Nyblom unstable | Regime split or rolling refit |


====================================================================================================
END OF FLATTENED SOURCE — examples/ + docs/ + docker/ + root config
====================================================================================================
```

---

## What's Now Complete

With these 21 files, the entire `volatility-mcp/` project is fully specified:

| Folder | Files | Status |
|---|---|---|
| `mcp_server/` | 35 | ✅ Complete (previous delivery) |
| `core/` | 104 | ✅ Complete (previous delivery) |
| `session_store/` | 4 | ✅ Complete (previous delivery) |
| `tests/` | 22 | ✅ Complete (previous delivery) |
| `examples/` | 4 | ✅ Complete (this delivery) |
| `docs/` | 8 | ✅ Complete (this delivery) |
| `docker/` | 3 | ✅ Complete (this delivery) |
| Root config | 6 | ✅ Complete (this delivery) |
| **Total** | **206** | **✅ Full project** |

## How to Restore

Save the block above as `examples_docs_docker_flattened.txt` and run `restore_flattened.py` against it. Combined with the previous three deliveries, the entire project tree will be reconstructed.

## How to Run

```bash
# 1. Restore all files
python restore_flattened.py

# 2. Install
make install-dev

# 3. Test
make test

# 4. Run examples
python examples/01_basic_flow.py
python examples/02_feedback_loop_demo.py
python examples/03_optimization_walkthrough.py

# 5. Start the MCP server
make run              # stdio (for Claude Desktop)
make run-sse          # SSE (for remote access)

# 6. Docker
make docker-build
make docker-up
