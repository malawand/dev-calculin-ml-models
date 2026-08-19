# BTC Cortex Feature Pipeline

A four-stage pipeline that pulls binary transition features from a Cortex/Prometheus
backend, aligns them with BTCUSDT hourly prices from ClickHouse, and measures how the
price behaved after each feature combination.

---

## Pipeline overview

```
fetch_cortex_metrics.py
        │  btc_cortex_features.csv
        ├──────────────────────────────► truth_table_cortex.py   (coverage check)
        ▼
align_features_with_last_price.py
        │  cortex_features_with_prices.csv
        ▼
count_moves.py                                                   (UP/DOWN/FLAT stats)
```

Run them in this order. Stages 2–4 all depend on the CSV produced by stage 1.

### The three features

Every script works with the same three binary (0/1) metrics:

| Column | Prometheus metric |
|---|---|
| `7d_derivative` | `derivative_transitions_2{interval="7d",order="99"}` |
| `7d_derivative_prime` | `derivative_prime_transitions_2{interval="7d_prime24h",order="99"}` |
| `poly_transition(_weighted24h48h7d)` | `polynomial_transition{interval="_weighted24h48h7d",order="90"}` |

Three binary features means 8 possible states, written as 3-digit strings (`000` … `111`)
in the digit order listed above.

---

## Requirements

```bash
pip install requests pandas clickhouse-connect
```

Python 3.8+. You also need network access to the Cortex endpoint and the ClickHouse
server — both are on internal IPs, so this only runs from inside the network (or over
VPN).

---

## 1. `fetch_cortex_metrics.py`

Pulls the three transition metrics out of Cortex over a multi-year window and writes them
to a single CSV on a continuous hourly grid.

**Input**

Nothing on the command line — everything is configured in the `CONFIGURATION` block at the
top of the file:

| Constant | Default | Meaning |
|---|---|---|
| `CORTEX_URL` | `http://10.1.20.60:9009/prometheus/api/v1/query_range` | Cortex range-query endpoint |
| `HEADERS` | `{}` | Auth headers, if your Cortex needs them |
| `OUTPUT_FILE` | `btc_cortex_features.csv` | Where the result is written |
| `QUERIES` | 3 metrics | Column name → PromQL string |
| `START_TIME_STR` | `2023-03-24T00:00:00Z` | Start of the range |
| `END_TIME_STR` | `2026-08-03T00:00:00Z` | End of the range |
| `STEP` | `1h` | Sample resolution |
| `CHUNK_DAYS` | `180` | Days per API request |

**Output**

`btc_cortex_features.csv` with columns:

```
time_bucket, 7d_derivative, 7d_derivative_prime, poly_transition(_weighted24h48h7d)
```

`time_bucket` is a UTC timestamp. Progress and a final row count are printed to stdout.

**Run**

```bash
python fetch_cortex_metrics.py
```

No flags. To change the date range, the metrics, or the output filename, edit the
constants at the top of the file.

**How it works**

- Each PromQL query is wrapped in `max()` so results aggregate across all instances/pods
  into one series.
- The range is split into 180-day chunks so no single request times out or blows past
  Cortex's point limit. Each chunk gets a 30-second timeout; a non-200 response or a
  connection error raises and stops the run.
- Per-metric frames are outer-merged on the raw UNIX timestamp, then reindexed onto a
  perfect hourly grid spanning the full range.
- Gaps are forward-filled, then **any row still holding a NaN is dropped** — `bfill` is
  deliberately not used. In practice this trims the leading period before all three
  metrics exist, so the output usually starts well after `START_TIME_STR`. That's
  expected: the file only contains rows where all three features are present.

---

## 2. `align_features_with_last_price.py`

Joins the Cortex features against hourly BTCUSDT closes from ClickHouse.

**Input**

- `btc_cortex_features.csv` (from stage 1), read from the working directory.
- A ClickHouse connection, configured inline at the top of the script:

| Setting | Default |
|---|---|
| `host` | `10.1.19.249` |
| `port` | `8124` |
| `username` | `exporter` |
| `password` | *see security note below* |

The query reads `binance_ww.raw_ticks_binance`, filters to `symbol = 'BTCUSDT'` between
`2023-03-24` and `2026-08-03`, buckets ticks into 1-hour intervals, and takes
`argMax(last_price, event_time)` as the close for each bucket.

**Output**

`cortex_features_with_prices.csv` — the stage-1 columns plus a `close_price` column. A
preview of the first rows is printed at the end.

**Run**

```bash
python align_features_with_last_price.py
```

No flags. The date range in the SQL and the input/output filenames are edited in-place.

**Notes**

- Both `time_bucket` columns are coerced to timezone-aware UTC before merging, since
  ClickHouse and the CSV don't agree on tz representation by default.
- The merge is a **left** join on the features, so every feature row survives. Hours with
  no tick data in ClickHouse get a null `close_price` — `count_moves.py` skips those rows
  when it computes returns.
- ⚠️ **Security:** the ClickHouse password is currently hardcoded. Before committing,
  replace it with an environment variable:

  ```python
  import os
  password=os.environ["CLICKHOUSE_PASSWORD"]
  ```

  and export it at runtime (`export CLICKHOUSE_PASSWORD=...`). If a real password was ever
  committed, rotate it — removing it in a later commit does not remove it from git history.

---

## 3. `truth_table_cortex.py`

A coverage check. Enumerates all 8 possible feature combinations and reports how many
hours in the dataset actually landed in each one — so you can see which states are common,
which are rare, and which never occur at all.

**Input**

`btc_cortex_features.csv` (hardcoded as `file_path`). Prices are not needed, so this can
run straight after stage 1.

**Output**

Printed to stdout only — no file is written. One row per combination:

| Column | Meaning |
|---|---|
| the 3 feature columns | the 0/1 combination |
| `occurrence_count` | how many hours matched it |
| `ever_occurred` | `True` if the count is above zero |

**Run**

```bash
python truth_table_cortex.py
```

No flags. To analyse a different set of columns, edit the `features` list near the bottom.
The commented-out `rename` block is left in for the case where the poly column arrives
under a different name.

**Notes**

- The truth table is built with `itertools.product([0.0, 1.0], ...)`, so the observed
  values must be floats for the merge to match. That's how the CSV comes out of stage 1.
- Combinations that never occurred still appear, with a count of 0. That's the point of
  generating the full table rather than just grouping.

---

## 4. `count_moves.py`

The analysis step. For each feature combination, counts how often the price moved UP,
DOWN, or stayed FLAT over the following N hours.

**Input**

`cortex_features_with_prices.csv` (from stage 2), passed as a positional argument.

**Output**

A table printed to stdout — one row per state plus an `ALL` summary row, showing UP/DOWN/
FLAT counts, the total, and each as a percentage. States with no observations show dashes.
Nothing is written to disk; pipe to a file if you want to keep it.

**Run**

```bash
# defaults: 24-hour lookahead, 0.1% flat threshold
python count_moves.py cortex_features_with_prices.csv

# widen the flat band
python count_moves.py cortex_features_with_prices.csv --threshold 0.25

# short-horizon view
python count_moves.py cortex_features_with_prices.csv --hours 4 --threshold 0.5
```

**Options**

| Flag | Default | Meaning |
|---|---|---|
| *(positional)* | — | Path to the CSV. Required. |
| `--features` | the 3 standard columns | Space-separated feature columns. Any count works — the state string grows to match, and the table lists 2ⁿ rows. |
| `--hours` | `24` | Lookahead horizon, in hours |
| `--threshold` | `0.1` | Moves smaller than this percent count as FLAT |
| `--time-col` | `time_bucket` | Name of the timestamp column |
| `--price-col` | `close_price` | Name of the price column |

**How it works**

- Exits with a clear error listing available columns if any requested column is missing.
- Sorts by time, drops duplicate timestamps (keeping the last), then reindexes onto a
  continuous hourly grid so the shift is a true N-hour lookahead rather than N rows.
- The state label is the feature values cast to int and concatenated, e.g. `101`. Rows
  where any feature is null get no state and are excluded.
- Return is `(price[t+N] / price[t] - 1) * 100`. Classification: above `+threshold` is UP,
  below `-threshold` is DOWN, within the band is FLAT. Rows without a usable state or
  result — including the final N hours, which have no future price — are dropped before
  counting.

**Reading the results:** raise `--threshold` and FLAT absorbs more rows, which shrinks
both UP% and DOWN%. Compare each state's UP% against the `ALL` row rather than against
50%, and check the `TOTAL` column — a state with only a handful of observations isn't
telling you much, and `truth_table_cortex.py` will show you which states those are.

---

## Typical full run

```bash
python fetch_cortex_metrics.py                              # → btc_cortex_features.csv
python truth_table_cortex.py                                # sanity check coverage
python align_features_with_last_price.py                    # → cortex_features_with_prices.csv
python count_moves.py cortex_features_with_prices.csv --hours 24 --threshold 0.1
```

