# Vector v1 schema

Three labeled artifacts, **never mixed**:

| Artifact | `vector_source` | `vector_version` | Status |
|---|---|---|---|
| Cortex binary baseline | `cortex_transition` (baseline, not a vector) | `cortex_transition_baseline_v1` | Phase 1 report |
| Price OLS proxy | `price_ols_proxy` | `price_ols_proxy_v1` | Built |
| Cortex market motion | `cortex_market_motion` | `cortex_market_motion_v1` | Built from historical Cortex D (8h, not 7h) |

OLS on log close is **price momentum**. It is not Cortex derivative-curve motion and is **not** a fallback when Cortex D is missing.

Sampling: hourly UTC, asset BTCUSDT.

- OLS builder: `market_motion/vector_engine.py`
- Cortex builder: `market_motion/cortex_vector.py`

**Frozen timestamp rule:** the live vector uses confirmation time, not hindsight peak/trough time. See `docs/TIMESTAMP_RULE.md`. Cortex publication delay is **uncertain**; see `docs/CORTEX_TIMING.md`.

## Horizons

### `price_ols_proxy_v1`

| Name | OLS bars | D/Z/T | V/A |
|---|---:|---|---|
| 2h | 3 | yes (noisy 3-point slope) | **no** — finer than 1h bars required |
| 4h | 4 | yes | no (`< 8` bars) |
| 7h | 7 | yes | no |
| 12h | 12 | yes | yes |
| 24h | 24 | yes | yes |
| 48h | 48 | yes | yes |
| 7d | 168 | yes | yes |

Rule: `min_ols_bars_for_slope=3`, `min_ols_bars_for_va=8`. Short windows set `resolution_class=slope_only` and `va_stable=false`.

### `cortex_market_motion_v1`

D is the actual Cortex continuous series. There is no `deriv7h`. Vector v1 uses **8h** (`deriv8h`). OLS 7h is never copied in.

| Name | Cortex series | Horizon hours | D/Z/T/V/A |
|---|---|---:|---|
| 2h | `job:crypto_last_price:deriv2h{symbol="BTCUSDT"}` | 2 | yes (`resolution_class=cortex_source`) |
| 4h | `job:crypto_last_price:deriv4h{symbol="BTCUSDT"}` | 4 | yes |
| 8h | `job:crypto_last_price:deriv8h{symbol="BTCUSDT"}` | 8 | yes |
| 12h | `job:crypto_last_price:deriv12h{symbol="BTCUSDT"}` | 12 | yes |
| 24h | `job:crypto_last_price:deriv24h{symbol="BTCUSDT"}` | 24 | yes |
| 48h | `job:crypto_last_price:deriv48h{symbol="BTCUSDT"}` | 48 | yes |
| 7d | `job:crypto_last_price:deriv7d{symbol="BTCUSDT"}` | 168 | yes |

Identity columns also include `source_query_version=cortex_continuous_d_prod_a_preferred_v1`, `source_semantic_certainty=owner_confirmed_routing_same_feature`, `selected_value_rule=prod_a_preferred_v1`, and `feature_configuration_hash`. Prime/weighted series may sit on the raw tape; they are not allowlisted.

Per-horizon **diagnostic provenance** (not model inputs): `{h}_source_cluster`, `{h}_source_replica`, `{h}_n_candidates`, `{h}_source_conflict`, `{h}_source_value_disagree`, `{h}_source_boundary`, `{h}_source_segment_id`, `{h}_selected_value_rule`. Cluster/replica are owner-confirmed routing metadata for the same feature family. Routing-label changes do not reset Z/V/A/T or restart normalization warmup.

## Per-horizon columns (`{h}_*`)

| Column | Meaning |
|---|---|
| `transition_state_raw` | Persistent 0/1 from confirmed **peaks/troughs of D** on every horizon. Null until first confirmation |
| `transition_raw` | Alias of `transition_state_raw` (CSV only; **not** in the model allowlist) |
| `transition_direction` | +1 / −1 once a turn has been confirmed (CSV only; not in the allowlist) |
| `transition_event` | `TROUGH_CONFIRMED` / `PEAK_CONFIRMED` on the confirmation row only; `NONE` otherwise |
| `confirmation_time` | Last confirmation timestamp (null until the first) |
| `hours_since_confirmation` | Hours since that confirmation (canonical age). `hours_since_turn` is a CSV alias only |
| `velocity_transition_*` | Diagnostic turns of V when `va_stable`. Not canonical T. Not in the allowlist |
| `candidate_extremum_time` | Metadata from confirmation onward. Null before. **Not a model input** |
| `derivative_raw` | Trailing causal OLS slope of log close **or**, on `cortex_market_motion_v1`, the Cortex continuous D sample |
| `position_z` | robust z of D using **t−1** median/MAD |
| `velocity` / `acceleration` | Δz, Δv; NaN unless `va_stable` |
| `va_stable` | false when bars < 8 |
| `ols_n_obs` / `ols_window_bars` / `resolution_class` | sample rule |
| `source_present` / `source_age_hours` / `normalization_ready` / `feature_valid` / `invalidation_reason` | quality |

`feature_valid` does **not** require a confirmed turn. D/Z can be valid while state is still null.

Detector: `causal_d_extrema_delay2_prom1p0_durf0p25_v2`. Status: **POC-frozen** (fixed for this proof of concept; not the final detector). Canonical T is always a confirmed peak/trough of **D**. Confirmation delay is 2 hours. The **timestamp rule is frozen**. Velocity turns are stored separately and are not model inputs.

## Timestamps

| Column | Meaning |
|---|---|
| `observation_time_utc` | bar close t used as feature |
| `feature_available_time_utc` | t + `feature_available_lag_hours` (default 1). This is `state_effective_time` |
| `decision_time_utc` | available + `decision_latency_hours` (default 0) |
| `outcome_entry_time_utc` | first executable close (same as decision unless extra latency) |

Do not fill at t. Primary 24h outcome starts at `outcome_entry_time_utc`, not at the confirmation bar’s close.

Row-level: `detector_version`, `confirmation_delay_hours`. Detector v2 is **POC-frozen** (fixed for this proof of concept; not the final detector).

## Feature allowlist

Analogues and future models must use `feature_allowlist.select_model_feature_columns`. Canonical inputs per horizon: `derivative_raw`, `position_z`, `velocity`, `acceleration`, `transition_state_raw`, `hours_since_confirmation`. That selector never returns oracle columns, waiting-cost fields, candidate timestamps, confirmation events, direction/raw aliases, velocity-turn fields, outcomes, or source-provenance/conflict metadata (`source_cluster`, `source_replica`, `n_candidates`, `source_conflict`, `source_value_disagree`, `source_boundary`, `source_segment_id`, `selected_value_rule`) unless explicitly approved later.

The analogue POC (`mixed_l1_equal_weight_poc_v1`) uses only this allowlist. Distance never includes outcomes, timestamps, routing metadata, or quality flags. Cortex and OLS remain separate experiments.

## Research-only oracle

Hindsight extrema live in `oracle_extrema_research_only.csv`, produced by `market_motion/oracle.py`. They are not merged into the vector or `research_view.csv`.

## Batch vs prefix

Prefix replay must match the batch row at t for every live feature, including confirmation state/event, missing prices, warmup, and stale sources (tested).
