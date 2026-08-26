# Market-motion research

Charter: [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md). Occupancy combo-ranking scripts are **out of scope**.  
Timestamp rule: [`../docs/TIMESTAMP_RULE.md`](../docs/TIMESTAMP_RULE.md).

## Artifact IDs (never mix)

| ID | Meaning |
|---|---|
| `cortex_transition_baseline_v1` | Binary Cortex 0/1 baseline |
| `price_ols_proxy_v1` | Causal OLS on log close (momentum proxy) |
| `cortex_market_motion_v1` | Continuous Cortex D (8h, not 7h). Collapse `prod_a_preferred_v1`. |

OLS is not a fallback for missing Cortex derivatives.

Live vector state uses **confirmation time**, not hindsight peak/trough time. Hindsight extrema live only in `oracle_extrema_research_only.csv`. Analogues and models must use `feature_allowlist.select_model_feature_columns`. Cluster/replica are diagnostic provenance, not model inputs.

## Install

```bash
cd features_analysis
python3 -m pip install -r requirements-research.txt
cp .env.example .env   # set CLICKHOUSE_PASSWORD; never commit .env
```

## Commands

```bash
# Tests
.venv/bin/python -m pytest market_motion/tests -q

# Proxy vector + outcomes from a local hourly price CSV
.venv/bin/python -m market_motion --config configs/baseline.yaml --run-id tape dataset --prices btc_last_price.csv

# Cortex binary baseline from a previously fetched aligned CSV
.venv/bin/python -m market_motion --config configs/baseline.yaml --run-id base baseline --aligned-csv cortex_features_with_prices.csv

# Rebuild cortex_market_motion_v1 from the preserved raw query_range tape
.venv/bin/python -m market_motion --run-id dataset_cortex_market_motion_v1_prod_a_preferred cortex-dataset \
  --raw-csv artifacts/dataset_cortex_market_motion_v1_tape/cortex_continuous_derivatives_raw.csv \
  --prices btc_last_price.csv \
  --ols-vector artifacts/dataset_price_ols_proxy_v1_tape/vector_price_ols_proxy_v1.csv

# Inference: vector file only
.venv/bin/python -m market_motion infer --vector artifacts/.../vector_cortex_market_motion_v1.csv

# Representation-only detector sanity (does not score 24h returns)
.venv/bin/python -m market_motion --run-id representation_sanity_poc representation-sanity \
  --vector artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/vector_cortex_market_motion_v1.csv

# Analogue POC on the development period only (holdouts stay closed)
.venv/bin/python -m market_motion --run-id analogue_poc_cortex_prod_a_preferred analogue-poc \
  --vector artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/vector_cortex_market_motion_v1.csv \
  --outcomes artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/outcomes_v1.csv

# Development-only walk-forward CRPS/Brier (holdouts stay closed)
.venv/bin/python -m market_motion --run-id analogue_eval_cortex_prod_a_preferred analogue-eval \
  --vector artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/vector_cortex_market_motion_v1.csv \
  --outcomes artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/outcomes_v1.csv

# Exploratory 4h/8h/12h/24h (same matcher and query dates; 24h target not replaced)
.venv/bin/python -m market_motion --run-id analogue_eval_horizons_cortex_prod_a_preferred analogue-eval-horizons \
  --vector artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/vector_cortex_market_motion_v1.csv \
  --outcomes artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/outcomes_v1.csv
```

Do not load `research_view.csv` or `oracle_extrema_research_only.csv` at inference. Do not overwrite `artifacts/dataset_cortex_market_motion_v1_tape/` (preserved segmented audit tape).

## Status

| Phase | Status |
|---|---|
| 0–3 | Code + tests. Confirmation-time frozen. Detector v2 **POC-frozen**. Source collapse `prod_a_preferred_v1`. **Program not frozen.** |
| Gate 0 | Password rotation **WAIVED/ACCEPTED RISK**. Cortex generator formula unknown. Cluster/replica are owner-confirmed routing metadata. |
| 4 | First analogue POC plus development-only walk-forward. Holdouts closed. Not Gate 4. |
| 5–8 | Not started. No LightGBM, feature selection, trading policy, or backtest |

Primary 24h target: approved 2026-08-24 (`docs/TARGET_FREEZE.md`). Locked-test dates still open.
