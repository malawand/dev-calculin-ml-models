# Implementation status (changing)

**Date:** 2026-08-25  
**Branch:** `market_motion_vector`  
**Charter:** `features_analysis/PROJECT_PLAN.md` v0.3 (full document). This file is status only.

The confirmation-time timestamp rule is frozen. Detector v2 is **POC-frozen** (not the final detector). Phase 4 analogue matching exists as a provisional POC on the development period only. Neither holdout was opened or scored.

## Artifact identity

| Artifact | `vector_source` | `vector_version` | Status |
|---|---|---|---|
| Cortex binary baseline | (baseline table, not a vector) | `cortex_transition_baseline_v1` | Completed earlier |
| Price OLS proxy | `price_ols_proxy` | `price_ols_proxy_v1` | Completed; separate momentum baseline |
| Cortex market motion (current) | `cortex_market_motion` | `cortex_market_motion_v1` | `prod_a_preferred_v1` tape |
| Cortex market motion (audit only) | `cortex_market_motion` | `cortex_market_motion_v1` | Preserved segmented `agree_or_conflict_v1` tape |

These must never be concatenated. OLS is never a fallback when Cortex D is missing.

**Current run:** `artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/`  
**Preserved segmented audit tape:** `artifacts/dataset_cortex_market_motion_v1_tape/` (do not overwrite)

`source_query_version`: `cortex_continuous_d_prod_a_preferred_v1`  
`selected_value_rule`: `prod_a_preferred_v1`  
`source_semantic_certainty`: `owner_confirmed_routing_same_feature`  
`feature_configuration_hash`: `4fc98dfd39e91e7ede5c435541fda46da3f793cd03243bfd9d7803bccc6bf47f`

## Frozen / versioned

| Item | Status |
|---|---|
| Confirmation-time timestamp rule | Frozen |
| Persistent `transition_state_raw` vs one-row `transition_event` | Frozen |
| Primary target: next-24h continuous BTC return | Approved 2026-08-24 |
| Canonical T | Confirmed turn of **D** |
| Detector v2 | **POC-frozen** — fixed for this proof of concept, not the final detector |
| ClickHouse password rotation | WAIVED/ACCEPTED RISK |
| Signed Cortex D collapse | `prod_a_preferred_v1` (never `max()`) |
| Source-semantic certainty | Owner-confirmed: cluster/replica are **routing metadata** for the same feature family. Upstream formula text is still not in this repo. |
| Cortex availability lag | `+1h` accepted provisionally; `timing_certainty=uncertain_recording_rule` |

## Cortex Vector v1 horizons

Original/general charter example includes **7h**. Live Cortex has no `deriv7h`. This version uses **8h**:

```text
2h, 4h, 8h, 12h, 24h, 48h, 7d
```

## Source-selection policy (`prod_a_preferred_v1`)

Owner clarification: Cortex `cluster` / `replica` identify where the series was delivered from/to Cortex. They are not different derivative formulas.

Live/future fetch injects `cluster="PROD-A-crypto-metrics"` into PromQL (`cortex_motion.live_cluster`). Historical reconstruction uses the preserved unfiltered `query_range` tape plus this priority:

```text
PROD-A-crypto-metrics
PROD-B-crypto-metrics
PROD-crypto-metrics
prod-calculin-tokyo-binance-scraper
prod-prom-cluster-1
DEV-crypto-metrics
unlabeled
```

`DEV-calculin-tokyo-binance-scraper-A` (1 raw row, identical to tokyo at 2023-12-30 19:00) is **not** on the list. It never wins; the overlapping tokyo row is selected. The priority was not changed.

At an overlapping timestamp: select the highest-priority **approved** source; record competing values in diagnostics; do **not** mark the hour missing when an approved source is present. True `source_conflict` only when no approved source exists (0 hours on Vector v1 horizons). Routing-label changes are provenance, not gaps: Z/V/A/T and the detector are continuous; normalization warmup is **not** restarted.

### Historical source selected (48h; 2h–24h are the same family)

| Selected cluster | Hours |
|---:|---:|
| `PROD-A-crypto-metrics` | 9,745 |
| `PROD-crypto-metrics` | 8,363 |
| `prod-calculin-tokyo-binance-scraper` | 5,197 |
| `PROD-B-crypto-metrics` | 3,032 |
| `prod-prom-cluster-1` | 2,591 |
| `DEV-crypto-metrics` | 657 |
| `unlabeled` | 14 |

PROD-A begins **2025-03-03 17:00 UTC**. Strict PROD-A-only history would omit everything before that and would contain a **3,017 hour** internal gap after **2025-06-27 05:00 UTC** (PROD-B covers through **2025-10-30 21:00**, then PROD-A resumes **2025-10-30 22:00**). Fallback keeps those hours; training/burn-in periods remain usable.

Same-timestamp overlaps: 4–6 hours/horizon; 2–3 value disagreements/horizon (still selected, not missing); sign disagreements at 2023-07-06 15:00 on 12h/24h (PROD-crypto preferred). True conflicts: **0**.

## Before / after usable history

Tape length **29,865**. First all-horizon valid row remains **2023-08-12 03:00 UTC**. Proposed burn-in through **2023-10-31** remains usable.

| Metric | Segmented `agree_or_conflict_v1` (preserved) | `prod_a_preferred_v1` (current) |
|---|---:|---:|
| All-horizon valid rows | 22,744 | **26,299** |
| Confirmed D turns | 22,224 | **22,649** |
| Normalization warmup (2h–48h) | ~4,196 / horizon | **335** / horizon |
| `source_boundary` invalidations | 18 / horizon | **0** (18 flags remain diagnostic; all 18 are feature-valid) |
| True source conflicts | 2–3 / horizon | **0** |

Routing no longer causes detector resets or a second Z warmup. 2h turns are 6,301 vs 6,303 on the earlier `max()` glue (two overlap hours now follow priority, not max). Other horizons match that earlier turn table (4h 6,087 … 7d 274).

## Date split (neither holdout scored)

| Window | Role |
|---|---|
| Through `2023-10-31` | Burn-in (not used for analogue matching) |
| `2023-11-01` → `2026-01-14` | Approved development / analogue POC |
| `2026-01-15` → `2026-06-30` | Retrospective outcome holdout — **closed** |
| No earlier than `2026-09-01` | Genuine prospective locked test — **closed** |

## Deferred Cortex ingestion debt (do not implement in this POC)

The eventual research-quality dataset will come from ClickHouse. Do not keep hardening the temporary Cortex tape.

1. **Historical vs live retrieval** are not fully separated. Live PromQL currently injects `cluster="PROD-A-crypto-metrics"`; historical rebuilds still use the preserved unfiltered `query_range` tape plus `prod_a_preferred_v1`. A dedicated live-query vs history-query split is deferred.
2. **Cluster priority is not configuration-driven at runtime.** The owner list is hardcoded in `market_motion/source_collapse.py` (`CLUSTER_PRIORITY`) and duplicated in `configs/baseline.yaml` for documentation. YAML does not currently drive selection. A fully configuration-driven priority system is deferred.

The current preserved raw tape and `prod_a_preferred_v1` artifact are sufficient for provisional experimentation.

## Detector v2 representation review

Lightweight, representation-only (no 24h scoring). Four random valid weeks; sampled confirmed turns on every horizon; stale/missing-D hours; confirmation timestamps.

- T changes at confirmation time: event row `confirmation_time` equals the row clock; `hours_since_confirmation=0`; candidate extremum is 2h earlier.
- Short vs long: median hours between turns 2h 4, 4h 3, 8h 6, 12h 8, 24h 16, 48h 28, 7d 85. 2h is noisier than 4h; 2h/8h still clearly faster than 24h/7d.
- Missing data: **0** confirmed turns on stale/missing Cortex D. Invalid-row events are warmup, not scrape holes.
- Status: **POC-frozen**. Artifact: `artifacts/representation_sanity_poc/representation_sanity.json`.

## Analogue POC (Phase 4, development only)

Label: **PROVISIONAL POC — temporary Cortex-derived dataset**

Distance `mixed_l1_equal_weight_poc_v1`: equal-weight mixed L1 on the explicit model-feature allowlist. Binary T mismatch plus robust median/MAD-scaled L1 of continuous columns, scaled from eligible **past** candidates only. Candidates must be strictly earlier than the query and their 24h outcome window must end before the query hour. Holdout rows are never queries or candidates. Cortex and OLS are separate runs.

| Experiment | Eligible 24h labels | Queries | Abstain | Uncond. median % | Uncond. P(+) | Analogue median-of-medians % | Analogue mean P(+) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cortex `prod_a_preferred_v1` | 19,017 | 45 | 0 | 0.12 | 0.530 | 0.16 | 0.549 |
| OLS `price_ols_proxy_v1` | 18,096 | 45 | 0 | 0.11 | 0.528 | 0.05 | 0.524 |

The first analogue cards compared batch medians to the full-sample BTC distribution. That is **not** the usefulness test. A later development-only walk-forward scored actual next-24h returns. A final exploratory pass repeated that walk-forward for 4h/8h/12h/24h without changing the matcher. See `docs/POC_RESULTS.md`.

Artifacts: `artifacts/analogue_poc_cortex_prod_a_preferred/`, `artifacts/analogue_poc_ols_proxy/`.

## Analogue walk-forward (development only)

Label:

```text
PROVISIONAL DEVELOPMENT-ONLY POC
Not locked evidence
Temporary Cortex-derived dataset
```

Fixed daily queries at 00:00 UTC after ~41 days of history (`2023-12-14` Cortex / `2023-12-21` OLS through `2026-01-13`). Matcher unchanged (`k=40`, 24h separation, past-only median/MAD). Past-only unconditional uses the same purged candidate pool. Holdouts closed. Abstain **0** on both runs.

| Experiment | n | Mean CRPS analogue | Mean CRPS uncond | Mean CRPS Δ | Beat rate | Brier Δ | Hit-rate follows predicted P |
|---|---:|---:|---:|---:|---:|---:|---|
| Cortex | 751 | 1.333 | 1.320 | +0.013 | 48.9% | +0.0009 | Yes (0.465 → 0.524 → 0.543) |
| OLS | 724 | 1.319 | 1.303 | +0.016 | 46.3% | +0.0008 | No (0.452 → 0.541 → 0.507) |

Cortex predicted-P tertiles also had mean 24h returns −0.06%, +0.35%, +0.26% (not fully ordered). Spearman predicted P vs actual return ≈ 0.06 on both. Closer analogue sets had **worse** CRPS skill than distant sets; a radius/abstention experiment is **not** recommended from this pass. Matcher was not tuned on these scores.

Artifacts: `artifacts/analogue_eval_cortex_prod_a_preferred/`, `artifacts/analogue_eval_ols_proxy/`.

## Remaining blockers

1. Upstream Cortex derivative **formula text** is still not in this repo.
2. 7d Cortex D does not exist before 2023-07-29.
3. Detector v2 is only POC-frozen; a later ClickHouse dataset needs a fresh review.
4. Prospective locked test reserved (`≥ 2026-09-01`) but not opened.
5. Password rotation remains an accepted risk.
6. Historical/live retrieval split and config-driven priority are deferred.
7. LightGBM, feature selection, trading policy, and trading backtests are not started. ClickHouse rebuild: `docs/CLICKHOUSE_REBUILD.md`.

## Commands

```text
cd features_analysis
.venv/bin/python -m pytest market_motion/tests -q
.venv/bin/python -m market_motion --run-id representation_sanity_poc representation-sanity \
  --vector artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/vector_cortex_market_motion_v1.csv
.venv/bin/python -m market_motion --run-id analogue_poc_cortex_prod_a_preferred analogue-poc \
  --vector artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/vector_cortex_market_motion_v1.csv \
  --outcomes artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/outcomes_v1.csv
.venv/bin/python -m market_motion --run-id analogue_poc_ols_proxy analogue-poc \
  --vector artifacts/dataset_price_ols_proxy_v1_tape/vector_price_ols_proxy_v1.csv \
  --outcomes artifacts/dataset_price_ols_proxy_v1_tape/outcomes_v1.csv
.venv/bin/python -m market_motion --run-id analogue_eval_horizons_cortex_prod_a_preferred analogue-eval-horizons \
  --vector artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/vector_cortex_market_motion_v1.csv \
  --outcomes artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/outcomes_v1.csv
```

Tests this pass: **101 passed**.
