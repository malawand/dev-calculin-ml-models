# Outcome v1 schema

`outcome_version`: `btc_outcomes_v1`  
Builder: `market_motion/outcome_engine.py`.  
`path_convention`: `hourly_close`.

Features at observation t use the close at t. They are **not** executable at that close.

Canonical returns start at `outcome_entry_time_utc` (default: next hourly close):

`return_k = (P[entry+k] / P[entry] - 1) * 100`

Also stored: `log_return_k`. Horizons: 1h, 2h, 4h, 8h, 12h, 24h, 48h.

UP/DOWN/FLAT is **derived**. Continuous return is canonical.

Incomplete windows are **NaN**, never 0. Missing future hours are not bridged.

## Close-path MFE/MAE (4h, 12h, 24h, 48h)

Named `close_path_mfe_*` / `close_path_mae_*` because they use **hourly closes only**. They underestimate intrahour extremes and are not tick-level executable excursions.

Sign: long-oriented (MFE = max path return, MAE = min path return) from the entry close.

Also stored: `time_to_close_path_mfe_*`, `time_to_close_path_mae_*` (hours after entry).

## Target freeze

**Approved 2026-08-24.** Primary: `return_24h` distribution from `outcome_entry_time_utc`, CRPS vs fold-unconditional. Secondary: P(R_24h > round-trip cost), Brier + calibration. See `docs/TARGET_FREEZE.md`. Locked-test dates are still open.


## Boundary

`outcomes_v1.csv` is research-only. Inference loads `vector_price_ols_proxy_v1.csv` only.
