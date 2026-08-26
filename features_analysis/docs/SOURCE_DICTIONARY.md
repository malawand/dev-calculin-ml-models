# Source dictionary (Phase 0)

Owner: research (`features_analysis`). Update when queries change. Hash queries in the run manifest.

## Cortex / Prometheus

Endpoint: `GET http://10.1.20.60:9009/prometheus/api/v1/query_range`  
Params: `query`, `start`, `end` (unix), `step`. Hourly: `step=1h`.

**Collapse is series-type specific.** Binary 0/1 transition queries in the table below wrap PromQL in `max()` because those bits are unsigned occupancy flags. **Signed continuous Cortex D must not use `max()` / `min()` collapse.** Vector v1 uses `prod_a_preferred_v1` (`market_motion/source_collapse.py`).

| Feature | PromQL | Unit / type | Cadence | Notes |
|---|---|---|---|---|
| `7d_derivative` | `max(derivative_transitions_2{interval="7d",order="99"})` | binary 0/1 | ~1h | 0 = upward after trough, 1 = downward after peak. **Not** sign(D). |
| `7d_derivative_prime` | `max(derivative_prime_transitions_2{interval="7d_prime24h",order="99"})` | binary 0/1 | ~1h | Same convention on the prime curve. |
| `poly_transition_weighted24h48h7d` | `max(polynomial_transition{interval="_weighted24h48h7d",order="90"})` | binary 0/1 | ~1h | Sparse; kept for baseline reproduction. |
| `close_price` (alt) | `max(crypto_last_price{symbol="BTCUSDT"})` | USD | 1h | OLS Vector v1 can use this instead of ClickHouse. |
| `volume` | `max(crypto_volume{symbol="BTCUSDT"})` | BTC | 1h | Not in Vector v1 minimum. |
| `deriv7d` continuous | `max(job:crypto_last_price:deriv7d{symbol="BTCUSDT"})` | source slope | 1h | Steepness context; not the 0/1 bit. |
| `weighted_deriv` | `max(job:crypto_last_price:weighted_deriv:24h:48h:7d{symbol="BTCUSDT"})` | source slope | 1h / 15m | Stage 4 nowcaster input; not Vector v1. |

**Causality:** Cortex *stored* 0/1 flags are treated as immutable after ingest (operator 2026-08-24). How the upstream generator first labels the hour is still unknown; see `docs/TRANSITION_CAUSALITY_AUDIT.md`. The **local** `price_ols_proxy_v1` vector uses confirmation time, not hindsight extrema (`docs/TIMESTAMP_RULE.md`).

## ClickHouse

| Field | Value |
|---|---|
| Host | `CLICKHOUSE_HOST` (default 10.1.19.249) |
| Port | `CLICKHOUSE_PORT` (default 8124) |
| User | `CLICKHOUSE_USER` (default exporter) |
| Password | **`CLICKHOUSE_PASSWORD` env only** |
| Table | `binance_ww.raw_ticks_binance` |
| Close | `argMax(last_price, event_time)` per UTC hour, `symbol='BTCUSDT'` |

A password was previously committed in `align_features_with_last_price.py`. It is removed from current code. **Rotate it on the server**; git history still contains it.

## Locally computed (`price_ols_proxy_v1` only)

This is a **separate hypothesis**, not a fill-in for missing Cortex D.

`derivative_raw` = causal rolling OLS slope of `log(close)` over the horizon bar count in `configs/baseline.yaml`. V/A from robust z of that slope, and only when the window has ≥ 8 hourly bars. Canonical T is a confirmed peak/trough of **D** on every horizon (`causal_d_extrema_*_v2`). Velocity turns are diagnostic only. See `docs/TIMESTAMP_RULE.md` and `docs/VECTOR_V1.md`.

## Continuous Cortex D (`cortex_market_motion_v1`)

Current run: `artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/`.  
Preserved segmented audit tape: `artifacts/dataset_cortex_market_motion_v1_tape/` (do not overwrite).  
Raw `query_range` dump is shared (`cortex_continuous_derivatives_raw.csv`, sha256 `87010b40…`). Instant query is not history.

| Vector horizon | PromQL (historical tape: unfiltered) | Live PromQL extra matcher | History from |
|---|---|---|---|
| 2h | `job:crypto_last_price:deriv2h{symbol="BTCUSDT"}` | `cluster="PROD-A-crypto-metrics"` | 2023-03-24 14:00Z |
| 4h | `job:crypto_last_price:deriv4h{symbol="BTCUSDT"}` | same | 2023-03-24 14:00Z |
| **8h** (not 7h) | `job:crypto_last_price:deriv8h{symbol="BTCUSDT"}` | same | 2023-03-24 14:00Z |
| 12h | `job:crypto_last_price:deriv12h{symbol="BTCUSDT"}` | same | 2023-03-24 14:00Z |
| 24h | `job:crypto_last_price:deriv24h{symbol="BTCUSDT"}` | same | 2023-03-24 14:00Z |
| 48h | `job:crypto_last_price:deriv48h{symbol="BTCUSDT"}` | same | 2023-03-24 14:00Z |
| 7d | `job:crypto_last_price:deriv7d{symbol="BTCUSDT"}` | same | **2023-07-29 00:00Z** |

**Missing:** `job:crypto_last_price:deriv7h`. Do not alias 8h as 7h. Do not fill with OLS.

Research-only (tape, not Vector v1 / allowlist):

| Name | PromQL | History from |
|---|---|---|
| `deriv7d_prime24h` | `job:crypto_last_price:deriv7d_prime24h{symbol="BTCUSDT"}` | 2025-04-30 |
| `deriv7d_prime7d` | `job:crypto_last_price:deriv7d_prime7d{symbol="BTCUSDT"}` | 2025-03-06 |
| `weighted_deriv_24h_48h_7d` | `job:crypto_last_price:weighted_deriv:24h:48h:7d{symbol="BTCUSDT"}` | 2025-03-03 |

### Routing vs formula

Owner-confirmed: Cortex `cluster` and `replica` identify **where** the series was delivered from/to Cortex. They are not different derivative formulas and do not change feature meaning. `source_semantic_certainty=owner_confirmed_routing_same_feature`. The upstream recording-rule / generator **formula text** is still not in this repository; that remains a separate documentation gap.

Canonical live source: `PROD-A-crypto-metrics` (`cortex_motion.live_cluster`). Future `query_range` injects that matcher. Historical reconstruction uses the unfiltered raw tape plus `prod_a_preferred_v1`.

Documented historical priority (do not reorder without reporting):

```text
PROD-A-crypto-metrics
PROD-B-crypto-metrics
PROD-crypto-metrics
prod-calculin-tokyo-binance-scraper
prod-prom-cluster-1
DEV-crypto-metrics
unlabeled
```

`DEV-calculin-tokyo-binance-scraper-A` appears once on the raw tape (2023-12-30 19:00, identical to tokyo). It is **not** on the list and never wins.

At an overlapping timestamp: select the highest-priority **approved** finite source; record competing values and `source_value_disagree`; do **not** mark the hour missing when an approved source is present. True `source_conflict` only when no approved source exists. Routing-label changes are provenance, not gaps: Z/V/A/T and the detector are continuous; normalization warmup is not restarted. Cluster/replica and the other `source_*` fields are diagnostic only (not model/analogue inputs).

Ordinary missing hours still use bounded 2h forward-fill after the raw save. A real time gap remains a gap. True conflicts stay missing and are not forward-filled. No backward-fill.

Audit: `artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/source_lineage_audit.md`. Timing: `docs/CORTEX_TIMING.md`. Do not mix Cortex T with OLS T.

## Deferred Cortex ingestion debt (POC)

Do **not** implement these during the current proof of concept. The research-quality dataset will come from ClickHouse.

1. Historical vs live retrieval are not fully separated. Live fetch injects `cluster="PROD-A-crypto-metrics"`; history uses the preserved unfiltered tape plus `prod_a_preferred_v1`.
2. Priority is hardcoded in `market_motion/source_collapse.CLUSTER_PRIORITY`. `configs/baseline.yaml` documents the same list but does not drive runtime selection.

