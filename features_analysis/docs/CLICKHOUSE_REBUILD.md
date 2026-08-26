# ClickHouse rebuild checklist

The current Cortex `query_range` tape and `prod_a_preferred_v1` vector are a **temporary POC source**. Do not keep hardening Cortex ingestion. Rebuild these layers on the cleaned ClickHouse dataset before any locked evaluation.

```text
EXPLORATORY DEVELOPMENT-ONLY POC
Not locked evidence
Temporary Cortex-derived dataset
```

## Do not carry forward as-is

| POC item | Status on Cortex tape | On ClickHouse |
|---|---|---|
| Detector v2 `POC-frozen` | Fixed for this proof of concept only | Repeat a representation-only review; do not treat POC-frozen as the final detector |
| `mixed_l1_equal_weight_poc_v1` | k=40, equal weights, no radius | Recompute distances on the new vector; do not copy POC skill scores |
| Development walk-forward scores | In-sample, overlapping daily 24h windows | Re-run with the same *rules*, not the same numbers |
| 8h analogue ranking (Cortex) | Exploratory; did not beat unconditional CRPS | Optional sensitivity only; 24h stays the approved primary target |
| Cortex vs OLS | Separate experiments; no winner | Keep separate; do not concatenate |

## Rebuild in order

1. **Prices and timestamps** — hourly BTCUSDT closes from ClickHouse with the frozen confirmation-time / entry-lag rules (`docs/TIMESTAMP_RULE.md`, `docs/TARGET_FREEZE.md`).
2. **Computed features** — D/Z/V/A/T on the ClickHouse clock. Do not splice OLS into missing Cortex D (and do not assume Cortex series still exist as the research source).
3. **Source selection** — drop PROD-A fallback machinery unless still required. Historical vs live retrieval split and YAML-driven cluster priority remain deferred Cortex debt; they should not block a ClickHouse rebuild.
4. **Vector identity** — new `vector_source` / `vector_version` / config hash. Mixed Cortex+OLS frames stay forbidden.
5. **Outcomes** — `return_4h`, `return_8h`, `return_12h`, `return_24h` with incomplete tails as NaN. Purge lags stay `entry + horizon + 1` (6h / 10h / 14h / 26h).
6. **Development / holdout masks** — burn-in through `2023-10-31`; development `2023-11-01` → `< 2026-01-15`; retrospective holdout `2026-01-15`–`2026-06-30` **closed** until a predeclared scoring pass; prospective locked test `≥ 2026-09-01` **closed**.
7. **Analogue index** — allowlist columns only; past-only median/MAD at each query; neighbours strictly earlier; outcome window finished before the query.
8. **Scores** — CRPS and Brier vs the *fold-specific* past-only unconditional. Do not compare raw CRPS across horizons. Do not reuse Cortex-tape skill numbers as evidence.
9. **Representation check** — T at confirmation time; short horizons faster than long; missing data does not invent turns.

## Explicitly out of scope until asked

- LightGBM / feature selection
- Trading-policy design
- Trading backtest
- Opening either holdout
- Tuning k, radius, or weights toward 8h because the Cortex ranking looked slightly sharper there

## Reference POC artifacts (do not treat as production)

- Vector: `artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/`
- Audit tape: `artifacts/dataset_cortex_market_motion_v1_tape/`
- Horizon eval: `artifacts/analogue_eval_horizons_cortex_prod_a_preferred/`, `artifacts/analogue_eval_horizons_ols_proxy/`
- Narrative: `docs/POC_RESULTS.md`
