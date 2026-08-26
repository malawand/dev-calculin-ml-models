# Temporary-data POC results

```text
EXPLORATORY DEVELOPMENT-ONLY POC
Multiple outcome horizons examined
Temporary Cortex-derived dataset
Not locked evidence
```

**Date:** 2026-08-25  
**Branch:** `market_motion_vector`  
**Approved primary target:** next-24h continuous BTC return (`return_24h`). That target is **not** replaced by this pass.

This file closes the temporary Cortex-derived analogue POC. It is not locked evidence. The research-quality dataset will come from ClickHouse.

## What was held fixed

Detector v2 remains `POC-frozen` (`causal_d_extrema_delay2_prom1p0_durf0p25_v2`). Matcher remains `mixed_l1_equal_weight_poc_v1` (allowlist only, k=40, 24h neighbour separation, past-only median/MAD). Daily queries remain 00:00 UTC on the 24h walk-forward calendar. Cortex and OLS were scored as separate experiments. k, weights, distance, detector, query timestamps, and features were not retuned per horizon.

## Outcome-window purge (tested)

With 1h entry lag and a strict “finished before the query bar” rule:

| Outcome | Candidate cutoff | Lag hours |
|---|---|---:|
| `return_4h` | `t <= query t − 6h` | 6 |
| `return_8h` | `t <= query t − 10h` | 10 |
| `return_12h` | `t <= query t − 14h` | 14 |
| `return_24h` | `t <= query t − 26h` | 26 |

Formula: `entry_lag + outcome_horizon + 1`. The 24h purge was not reused for shorter horizons. Tests: `market_motion/tests/test_analogue_eval.py`.

## Shared schedule

| Vector | First query (UTC) | Last query (UTC) | n scored | Abstain |
|---|---|---|---:|---:|
| Cortex `prod_a_preferred_v1` | 2023-12-14 00:00 | 2026-01-13 00:00 | 751 / horizon | 0 |
| OLS `price_ols_proxy_v1` | 2023-12-21 00:00 | 2026-01-13 00:00 | 724 / horizon | 0 |

Holdouts (`2026-01-15`–`2026-06-30` and `≥ 2026-09-01`) were not opened.

Do not compare raw CRPS across 4h/8h/12h/24h. Use CRPS difference and CRPS skill `1 − analogue_CRPS / unconditional_CRPS`. Positive skill means the analogue beat that horizon’s own past-only unconditional.

## Comparison (development only)

| Outcome horizon | Cortex CRPS skill | Cortex Brier skill | Ranking monotonic? | OLS CRPS skill | Interpretation |
|---|---:|---:|---|---:|---|
| 4h | −0.016 | −0.023 | No (realized 48.0% → 49.6% → 49.8%) | −0.008 | Weak. OLS 4h ranks slightly; analogue still does not beat the unconditional. |
| 8h | −0.021 | −0.012 | Yes (47.3% → 53.3% → 57.3%) | −0.020 | Clearest Cortex probability ranking; density forecast is worse, not a new target. |
| 12h | −0.022 | −0.017 | No | −0.023 | Weak at both sources. |
| 24h | −0.010 | −0.004 | Yes (46.5% → 52.4% → 54.3%) | −0.012 | Approved target. Small ranking, no CRPS skill. CI on Cortex CRPS Δ includes 0. |

“Ranking monotonic?” means realized P(return > 4 bps) rose across analogue predicted-P tertiles. Mean 24h returns were not strictly ordered on Cortex 8h/24h.

## Cortex detail

| Horizon | CRPS analogue | CRPS uncond | CRPS Δ | Beat % | CRPS Δ bootstrap 90% | Spearman P vs return |
|---|---:|---:|---:|---:|---|---:|
| 4h | 0.487 | 0.479 | +0.008 | 50.7% | +0.003 to +0.014 | −0.014 |
| 8h | 0.677 | 0.663 | +0.014 | 48.9% | +0.007 to +0.022 | 0.058 |
| 12h | 0.812 | 0.794 | +0.018 | 47.0% | +0.008 to +0.026 | 0.023 |
| 24h | 1.333 | 1.320 | +0.013 | 48.9% | −0.002 to +0.026 | 0.057 |

Closer analogue sets were not more useful than distant sets on CRPS at any Cortex horizon.

## OLS detail (separate experiment, not a winner)

| Horizon | CRPS analogue | CRPS uncond | CRPS Δ | Beat % | CRPS Δ bootstrap 90% | Spearman | Ranking monotonic? |
|---|---:|---:|---:|---:|---|---:|---|
| 4h | 0.480 | 0.477 | +0.004 | 51.2% | −0.001 to +0.009 | 0.068 | Yes (46.5% → 48.5% → 52.8%); mean return 0.00% → 0.02% → 0.04% |
| 8h | 0.675 | 0.662 | +0.013 | 48.1% | +0.006 to +0.020 | 0.029 | Hit-rate yes; mean return not ordered |
| 12h | 0.808 | 0.790 | +0.018 | 47.0% | +0.008 to +0.027 | 0.037 | No |
| 24h | 1.319 | 1.303 | +0.016 | 46.3% | +0.003 to +0.031 | 0.054 | No |

OLS 4h is the only OLS cell where closer neighbours had a slightly better CRPS than distant ones. The effect is small and was not used to change k or add a radius.

## Answers

1. **Clearer relationship at 4h/8h/12h than 24h?** No. Cortex 8h has a slightly wider exceed-cost ranking than 24h. 4h and 12h do not. No shorter horizon is a better distributional forecast.
2. **Does any horizon beat its own past-only unconditional?** No. Every CRPS skill and Brier skill is negative. Cortex 4h/8h/12h CRPS gaps are worse than chance on the bootstrap interval. Cortex 24h includes no-skill.
3. **Do low vs high predicted-P groups have different outcomes?** Sometimes, modestly. Cortex 8h (47% vs 57%) and Cortex 24h (46% vs 54%) show a low-to-high step. Cortex 4h is flat. Mean returns do not rise cleanly. This is not a trading signal.
4. **Stable enough to justify retesting with ClickHouse?** Only as a **sensitivity check**, not a retarget. If ClickHouse work happens, keep **24h as the approved primary** and may re-score 8h analogue ranking on the cleaned tape. That is a confirmation candidate, not a discovered winner.
5. **Evidence weak at every horizon?** Yes. The matcher produces different-looking probabilities; they do not beat a past-only BTC distribution, and they are not stable enough to open a holdout, start ML, or design a policy.

## What this pass did not do

- No change to k, radius, weights, detector, vector, or query dates.
- No LightGBM, feature selection, trading policy, or backtest.
- Neither holdout opened or scored.
- No Cortex vs OLS winner.

## Artifacts

- `artifacts/analogue_eval_horizons_cortex_prod_a_preferred/`
- `artifacts/analogue_eval_horizons_ols_proxy/`
- Prior 24h-only walk-forward (unchanged matcher): `artifacts/analogue_eval_cortex_prod_a_preferred/`, `artifacts/analogue_eval_ols_proxy/`

Rebuild steps after ClickHouse: `docs/CLICKHOUSE_REBUILD.md`.
