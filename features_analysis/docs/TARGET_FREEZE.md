# Target freeze (approved for development)

**Status: approved 2026-08-24** (primary horizon: next 24 hours).  
This is not a locked-test date freeze. Analogue matching was scored as a **provisional development-only walk-forward**, including an exploratory 4h/8h/12h comparison. The approved primary target remains next-24h. Do not open either holdout.

Do **not** pick the target by searching for the best-performing horizon.

| Role | Choice |
|---|---|
| Primary horizon | next 24 hours |
| Primary target | continuous BTC return distribution (`return_24h` from `outcome_entry_time_utc`) |
| Primary score | CRPS vs the **fold-specific** unconditional distribution |
| Secondary target | \(P(R_{24h} > \) estimated round-trip cost\()\) |
| Secondary score | Brier + calibration |
| Supporting | median, quantiles, `close_path_mfe` / `close_path_mae`, effective independent analogue support |

Round-trip cost default: `0.04` percent (4 bps). Revisit before any locked test.

## Vector sources (different hypotheses)

| Artifact | What \(D\) is | Version |
|---|---|---|
| `price_ols_proxy` | Causal OLS slope of log close (momentum) | `price_ols_proxy_v1` |
| `cortex_market_motion` | Cortex derivative curves around peaks/troughs | `cortex_market_motion_v1` (built; 8h not 7h; `prod_a_preferred_v1`) |
| `cortex_transition_baseline` | Binary 0/1 Cortex transitions | `cortex_transition_baseline_v1` |

Never concatenate them.

## Execution

Close at t is not tradable at t. Default: `outcome_entry_time_utc = t + 1h`.
