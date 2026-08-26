# Gate 0–3 report

**Branch:** `market_motion_vector`  
**Date:** 2026-08-25  
**Rule:** code existing ≠ gate passed.

This pass keeps the **confirmation-time** rule frozen (`docs/TIMESTAMP_RULE.md`). Detector v2 is **POC-frozen** (not the final detector). Canonical T is a confirmed peak/trough of **D**. A first analogue POC ran on the development period only. ML and trading policy are not started.

Project Plan document version is **0.3** (full charter). Changing status lives in `docs/IMPLEMENTATION_STATUS.md`.

## Separate vector statuses

| Artifact | Status | Notes |
|---|---|---|
| `price_ols_proxy_v1` | Completed separate momentum baseline | 29,865 rows; detector v2 on OLS D; do not replace with Cortex |
| `cortex_market_motion_v1` (current) | Built; `prod_a_preferred_v1` | New run `artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/`; 8h not 7h; no OLS fallback |
| `cortex_market_motion_v1` (audit) | Preserved | Segmented `agree_or_conflict_v1` tape at `artifacts/dataset_cortex_market_motion_v1_tape/` |
| Cortex source-timing certainty | UNCERTAIN | Conservative 1h lag; recording-rule eval time unproven (`docs/CORTEX_TIMING.md`) |
| Cluster/replica semantics | Owner-confirmed routing | Same feature family; formula text still not in this repo |
| Historical-tape coverage | PARTIAL | 2h–48h from 2023-03-24 14:00Z; **7d from 2023-07-29**; primes from 2025 |

## Gate 0 — Security / source audit

| Requirement | Status | Evidence |
|---|---|---|
| ClickHouse password not in current source | PASS | `align_features_with_last_price.py`, `market_motion/prices.py` |
| Password rotation confirmed | WAIVED/ACCEPTED RISK | Dev environment. A previously committed password remains in git history |
| Secrets not printed/committed this pass | PASS | `.env` gitignored; fetch logs have PromQL only |
| Source dictionary | PASS | `docs/SOURCE_DICTIONARY.md` |
| Cortex queries recorded | PASS | `configs/baseline.yaml` `cortex_motion.horizons`; `cortex_retrieval_log.json` |
| Peak/trough method known | BLOCKED for Cortex recording rules | Generator formula text still not in this repo. Local detector v2 is known and tested |
| Confirmation uses only past bars | PASS for local detector | `causal_d_extrema_*_v2` on both OLS D and Cortex D |
| Knowable time vs backdate | PASS for local detector | State changes at confirmation; 2:00/3:00 not rewritten |
| Historical Cortex immutable after ingest | PASS | Operator 2026-08-24 |
| Initial Cortex timestamp = publication time | NOT TESTED / UNCERTAIN | Immutability ≠ correct first timestamp (`docs/CORTEX_TIMING.md`) |
| OLS never silent Cortex fallback | PASS | OLS `build_vector` raises; Cortex builder raises `CortexSourceError` if a required series is missing |
| Signed Cortex D not collapsed with `max()` | PASS | `prod_a_preferred_v1`; `test_signed_disagreement_is_never_max`; `test_prod_a_wins_when_present` |
| Live fetch selects PROD-A | PASS | `apply_live_cluster_selector`; `test_live_fetch_selector_injects_prod_a` |
| Historical fallback is deterministic | PASS | `test_historical_fallback_when_prod_a_absent`; unapproved-only is a true conflict |
| Source lineage audited | PASS (routing confirmed) | `source_lineage_audit.json` / `.md`; cluster labels are delivery provenance |
| True source conflicts are hard missing | PASS | No ffill through true conflict; `test_true_conflict_is_not_forward_filled` |
| Routing labels do not reset Z/V/A/T | PASS | Continuous detector; `test_routing_label_change_does_not_reset_features_or_detector`; 18/18 8h boundary hours are feature-valid |
| Provenance excluded from model allowlist | PASS | `test_allowlist_excludes_provenance_metadata` |

**Gate 0 overall:** password rotation remains an accepted risk. Local confirmation-time rule is implemented on both vectors. Cortex *stored* values treated as immutable. Cortex *initial* timestamping is not proven. Cluster/replica are owner-confirmed routing metadata for the same feature family. The upstream derivative formula text is still unavailable.

## Gate 1 — Reproducible foundation

| Requirement | Status | Evidence |
|---|---|---|
| Hourly unique UTC grid | PASS | `test_reindex_is_hourly_unique_monotonic`, `test_utc_day_boundary_alignment` |
| Shift is hours not rows | PASS | `test_shift_hours_not_rows` |
| No bfill | PASS | `test_never_bfill_and_staleness`, `test_never_backward_fill_cortex_d` |
| Bounded source-specific ffill | PASS | `quality.bounded_ffill`; applied only after raw Cortex save; ordinary gaps only |
| Long outage does not stay valid | PASS | `test_long_outage_never_stays_valid`, `test_missing_hours_still_obey_staleness` |
| Quality flags | PASS | `{h}_source_age_hours`, `_source_present`, `_normalization_ready`, `_feature_valid`, `_invalidation_reason` |
| Manifests + checksums | PASS | OLS and Cortex `manifest.json`; raw sha256 `87010b4080925811b1b416e9c12361eb85794159864219cba5a695ebfa6b1f5d` |
| Prefix vs full history | PASS | OLS prefix tests + `test_batch_vs_prefix_parity_cortex` + `test_prefix_parity_with_provenance` |
| Historical Cortex `query_range` | PASS | 10 series, 90-day chunks, ~38.7s wall time, 241,183 raw rows |
| Live ClickHouse refetch | NOT TESTED | `CLICKHOUSE_PASSWORD` unset; local `btc_last_price.csv` reused |
| Inference loads vector only | PASS | research_view / oracle / mixed-source tests; Cortex vector loads |

**Gate 1 overall: FAIL** (tests pass; locked-test dates still open; ClickHouse live refetch not repeated; Cortex timing unproven)

## Gate 2 — Vector v1

Identity, named artifacts, mixed-source refusal, persistent T vs event, unknown initial state, Z from t−1, four timestamps, no same-close entry, allowlist, and no oracle columns: **PASS** (OLS and Cortex tests).

### `price_ols_proxy_v1`

Completed earlier this branch. Horizons 2h/4h/**7h**/12h/24h/48h/7d. Tape: 29,865 hourly rows, 2023-03-24 14:00Z–2026-08-19 22:00Z. All-horizon valid 27,383. 7d OLS turns: 278 events, median 88h between.

OLS engine still **raises** if asked to build Cortex. That is required.

### `cortex_market_motion_v1`

| Requirement | Status | Evidence |
|---|---|---|
| Built from actual Cortex D | PASS | `build_cortex_vector`; D columns are Cortex series |
| Horizons 2h/4h/**8h**/12h/24h/48h/7d | PASS | `test_cortex_horizons_are_eight_not_seven` |
| No 7h / no OLS 7h insert | PASS | `test_seven_h_horizon_rejected`; no `7h_` columns |
| Missing required series fails | PASS | `test_missing_required_cortex_source_fails`, `test_all_nan_cortex_source_fails_without_ols_fallback` |
| Canonical T from Cortex D | PASS | `test_canonical_t_from_cortex_d_not_velocity` |
| One T + one age in allowlist | PASS | `test_one_canonical_transition_and_age_per_horizon` |
| Confirmation-time / unknown initial | PASS | `test_unknown_initial_state_and_confirmation_time` |
| Source availability lag | PASS | `feature_available_lag_hours=1`; `timing_certainty=uncertain_recording_rule` |
| Historical coverage complete | FAIL | 7d missing until 2023-07-29 (leading hours); ~0.8–0.9% interior gaps on 2h–48h |
| Timing proven | FAIL | Conservative lag only |

Identity: `source_query_version=cortex_continuous_d_prod_a_preferred_v1`, `selected_value_rule=prod_a_preferred_v1`, `source_semantic_certainty=owner_confirmed_routing_same_feature`, `feature_configuration_hash=4fc98dfd39e91e7ede5c435541fda46da3f793cd03243bfd9d7803bccc6bf47f`.

Tape: 29,865 hourly rows (same price grid). All-horizon valid **26,299 / 29,865**. First all-horizon valid row remains **2023-08-12 03:00 UTC**. Detector: `causal_d_extrema_delay2_prom1p0_durf0p25_v2` (**POC-frozen**, not the final detector). Normalization warmup is a single series (**335** hours/horizon), not per routing segment.

| Horizon | Source present | Valid | Invalid | Valid frac | True conflicts | Boundary flags (all valid) |
|---|---:|---:|---:|---:|---:|---:|
| 2h | 29,615 | 29,316 | 549 | 0.982 | 0 | 18 |
| 4h | 29,620 | 29,320 | 545 | 0.982 | 0 | 18 |
| 8h | 29,593 | 29,293 | 572 | 0.981 | 0 | 18 |
| 12h | 29,621 | 29,321 | 544 | 0.982 | 0 | 18 |
| 24h | 29,593 | 29,293 | 572 | 0.981 | 0 | 18 |
| 48h | 29,599 | 29,297 | 568 | 0.981 | 0 | 18 |
| 7d | 26,623 | 26,312 | 3,553 | 0.881 | 0 | 15 |

Invalidation is ordinary: warmup, stale/missing Cortex D, and velocity undefined. `source_boundary` is **not** an invalidation reason.

### Cortex turn frequency (canonical D turns)

| Horizon | `max()` glue | Segmented `agree_or_conflict_v1` | `prod_a_preferred_v1` | Peak | Trough | Hours / event | Median hours between | Median hours since confirm | Median \|ΔD\| cand→confirm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2h | 6,303 | 6,166 | **6,301** | 3,150 | 3,151 | 4.7 | 4 | 2 | 0.070 |
| 4h | 6,087 | 5,966 | **6,087** | 3,043 | 3,044 | 4.9 | 3 | 2 | 0.036 |
| 8h | 4,212 | 4,133 | **4,212** | 2,106 | 2,106 | 7.1 | 6 | 3 | 0.011 |
| 12h | 3,123 | 3,064 | **3,123** | 1,561 | 1,562 | 9.6 | 8 | 4 | 0.0050 |
| 24h | 1,709 | 1,677 | **1,709** | 854 | 855 | 17.5 | 16 | 8 | 0.0013 |
| 48h | 943 | 938 | **943** | 471 | 472 | 31.7 | 28 | 16 | 0.00039 |
| 7d | 274 | 280 | **274** | 137 | 137 | 109.0 | 85 | 49 | 3.8e-5 |
| **total** | **22,651** | **22,224** | **22,649** | | | | | | |

2h is 2 events below the earlier `max()` glue because two overlap hours now follow priority instead of max. Other horizons match that earlier table. 7d no longer gains extra first-confirmations from segment resets. Confirmed turns on routing-boundary hours are **allowed** (4 on 2h; 0 on 8h and 7d). Detector was **not** tuned on next-24h returns.

### Cortex vs OLS (diagnostic only)

Do not concatenate. Do not pick a winner on the 24h target. The 8h-vs-7h and 7d T/D-sign table from the segmented tape was not re-opened this pass.

Spot-check plots/tables: `artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/spot_checks/`.

**Gate 2 overall: FAIL** (OLS baseline complete; Cortex vector rebuilt under `prod_a_preferred_v1`; historical 7d incomplete; Cortex timing unproven; detector is only POC-frozen; dates not frozen)

## Gate 3 — Outcome v1

| Requirement | Status | Evidence |
|---|---|---|
| Continuous returns canonical | PASS | `return_{k}h`, `log_return_{k}h` |
| Horizons 1,2,4,8,12,24,48h | PASS | `test_returns_all_horizons_present` |
| Entry after close | PASS | `test_no_same_close_execution_leakage`, `test_no_same_close_execution` |
| Unlabeled tails not zero | PASS | `test_unlabeled_tails_not_zero` |
| Close-path MFE/MAE naming | PASS | `close_path_mfe_*`, `close_path_mae_*` |
| 24h target approved | PASS | Approved 2026-08-24; `target_freeze.status: approved` |
| Cortex 24h outcome relationship | POC only | Development-only analogue walk-forward; holdouts not scored |
| CRPS / Brier scored | POC only | Walk-forward vs past-only unconditional; not a locked evaluation |

**Gate 3 overall:** engine tested; **24h target approved 2026-08-24**. Locked-test dates still open.

## Phase 4 — Analogue POC (not Gate 4)

A first analogue pipeline ran on development data only (`2023-11-01` → `2026-01-14`). Distance `mixed_l1_equal_weight_poc_v1`. Cortex and OLS were separate experiments. A later pass scored the **same matcher** with a daily walk-forward (CRPS vs past-only unconditional, Brier, calibration), then repeated it for 4h/8h/12h/24h. Results: `docs/POC_RESULTS.md`. Gate 4 is **not** passed. LightGBM, feature selection, trading policy, and backtests were not started. No horizon was tuned toward.

## Tests

```text
cd features_analysis
.venv/bin/python -m pytest market_motion/tests -q
# 101 passed
```

## Freeze recommendation

**Do not freeze the research program.** Detector v2 is **POC-frozen** only. Confirmation-time remains frozen. Development-only analogue walk-forward is **not** locked evidence. Do not score holdouts. Do not further harden Cortex ingestion; ClickHouse is the intended research dataset.
