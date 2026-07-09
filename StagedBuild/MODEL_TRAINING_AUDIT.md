# Model Training Audit — BTC Regime Nowcaster (Stage 4 LightGBM)

**Purpose of this document.** A complete, self-contained technical record of how
the Stage 4 model was trained, so an independent engineer or AI can audit it for
correctness, leakage, and live-readiness. All figures below were regenerated from
the committed artifacts (`4_Classifier/model_artifacts/`, the Stage 3 parquet, and
the Stage 6 backtest reports). Where a value comes from code, the file is cited.

> **TL;DR for auditors.** This is a **coincident multi-class classifier** (a
> nowcaster, horizon = 0 bars) that reproduces a **rule-based, price-derived
> regime label**. Every input feature and the label itself derive from a single
> BTC price series. Validation is a **single chronological split with no
> purge/embargo**. There are two identified leakage vectors (both small,
> documented in §10). The label uses **non-causal hysteresis**. No hyperparameter
> search was run. The downstream trade layer (Stage 6) produced **zero trades** in
> its committed backtest, so there is **no realized trading performance**.

---

## 1. Model objective

| Question | Answer |
|----------|--------|
| Task type | **Multi-class classification** (4 classes), not regression/ranking. |
| What it predicts | The **market regime of the current 15-minute bar**. |
| Target label | `regime ∈ {0: CHOP, 1: TRENDING_UP, 2: TRENDING_DOWN, 3: VOLATILE_EXPANSION}` |
| Prediction horizon | **0 bars (coincident / nowcast).** The model predicts the label *of bar t* from features known at bar t. It does **not** forecast a future bar. |
| Basis of target | **Log-return + realized-volatility thresholds** on price (see §2). Not fee-adjusted, not profitability-based. |

The model is defined in `4_Classifier/train.py`; objective is LightGBM
`multiclass` with `num_class = 4`.

Important nuance for auditors: because the horizon is 0 and the label is a
deterministic function of price (which is available live), the model is
*approximating a computable rule*, not forecasting the unknown. Its only possible
edge over the rule is smoothing / slightly-earlier detection at transitions.

---

## 2. Target construction

### 2.1 Source of the label

Labels are built in `2_Build_Labels/build_labels.py` from the `price` column only.
All rolling features used for labeling are **backward-looking**; the raw
classification is causal, but the **hysteresis post-processing is not** (see §2.4).

### 2.2 Exact formula

Bar constants (15-minute bars): `BARS_4H=16`, `BARS_12H=48`, `BARS_24H=96`,
`BARS_7D=672`. `ANNUALIZE_FACTOR = sqrt(4*24*365)`.

```python
log_price = np.log(price)
bar_logret = log_price.diff()

rolling_return_4h  = log_price - log_price.shift(16)
rolling_return_12h = log_price - log_price.shift(48)
rolling_return_24h = log_price - log_price.shift(96)
rolling_return_7d  = log_price - log_price.shift(672)

realized_volatility_4h  = bar_logret.rolling(16).std()  * ANNUALIZE_FACTOR
realized_volatility_24h = bar_logret.rolling(96).std()  * ANNUALIZE_FACTOR
```

Raw regime assignment (priority: VOLATILE_EXPANSION > TRENDING_UP > TRENDING_DOWN
> CHOP), with default thresholds:

```python
# DEFAULTS
trend_return_threshold       = 0.008   # 0.8% over 24h
trend_return_threshold_12h   = 0.006   # 0.6% over 12h
trend_volatility_ratio_max   = 2.0
volatile_return_threshold    = 0.01    # 1% over 4h
volatile_volatility_ratio    = 1.3

vol_ratio_ok = realized_volatility_4h < 2.0 * realized_volatility_24h

is_volatile      = (|rolling_return_4h| > 0.01) & (realized_volatility_4h > 1.3 * realized_volatility_24h)

is_trending_up   = ((rolling_return_24h > 0.008) & (rolling_return_7d >= 0) & vol_ratio_ok)
                 | ((rolling_return_12h > 0.006) & (rolling_return_24h > 0) & vol_ratio_ok)

is_trending_down = ((rolling_return_24h < -0.008) & (rolling_return_7d <= 0) & vol_ratio_ok)
                 | ((rolling_return_12h < -0.006) & (rolling_return_24h < 0) & vol_ratio_ok)

regime = 0 (CHOP)
regime[is_trending_down] = 2
regime[is_trending_up]   = 1   # overwrites down
regime[is_volatile]      = 3   # highest priority
```

### 2.3 Look-ahead in the label

- **Feature computation for the label:** backward-looking only (`shift(+N)`,
  trailing `rolling`). No `shift(-N)`.
- **Horizon:** none. The label describes bar t using data up to and including t.

### 2.4 Non-causal hysteresis (auditor flag)

After raw classification, `_apply_hysteresis(min_bars=4)` absorbs any regime run
shorter than 4 bars into the surrounding run. Determining that a run is "short"
requires seeing how it **ends**, so `regime[t]` can depend on **up to a few future
bars** (the merge loop can chain further). Consequence:

- The final training label near a transition is **not reproducible at time t live**.
- This is acceptable for *hindsight training targets* but means bar-level accuracy
  near transitions is measured against a slightly future-aware label.

### 2.5 Fees / spread / slippage / neutral labels

- **Fees, spread, slippage, minimum-profit threshold:** **none.** The label is
  purely statistical (returns + volatility), not profitability-based.
- **Neutral / no-trade class:** `CHOP` (class 0) is the de-facto neutral state.
  There is no separate "no-trade" label; CHOP is a genuine regime, not an abstain.

### 2.6 Class distribution

Full labeled dataset (`3_Momentum_State_Encoder/btc_data_15m_mstate.parquet`,
37,628 rows, before the classifier's warm-up drop):

| Regime | Count | Share |
|--------|-------|-------|
| CHOP (0) | 13,830 | 36.8% |
| TRENDING_UP (1) | 10,878 | 28.9% |
| TRENDING_DOWN (2) | 10,228 | 27.2% |
| VOLATILE_EXPANSION (3) | 2,692 | 7.2% |

Per-split distribution is in §5.3. Note the class balance **drifts** across splits
(CHOP 38.8% train → 27.2% test), which matters for the val→test gap.

---

## 3. Training data

| Attribute | Value |
|-----------|-------|
| Instrument | **BTCUSDT** (spot last-price). |
| Source | **Prometheus / Cortex** `query_range` API at `http://10.1.20.60:9009` (internal). Series are recording rules over `crypto_last_price{symbol="BTCUSDT"}`. |
| Underlying exchange | **Not specified in code** — the metric is `crypto_last_price`; the source exchange is not recorded in this repo. |
| Bar interval | **15 minutes** (`--step 900`). A 1-hour variant (`btc_data.parquet`) exists but was **not** used for this model. |
| Date range (encoded parquet) | **2025-03-18 00:15:18 UTC → 2026-04-15 00:15:18 UTC** |
| Date range (after warm-up drop, used for modeling) | **2025-03-22 00:15:18 UTC → 2026-04-15 00:15:18 UTC** |
| Rows (encoded parquet) | 37,628 |
| Rows used for modeling | **36,770** (858 warm-up rows dropped) |
| Duplicate timestamps | **0** |
| Time gaps | 2 gaps in the whole series (one 14:45, one 11:00 vs. the 15-min cadence); otherwise contiguous. |
| Stale data | `price` is a *last-price* sample; stale/held values between updates are possible and are **not** de-duplicated or flagged. |

### 3.1 Ingestion / export

`1_BTC_Data_Export/export_prometheus.py` pulls four series into a wide parquet:
`price`, `weighted_norm_avg_16h_24h_48h`, `weighted_deriv_24h_48h_7d`,
`norm_combined_avg`.

### 3.2 Missing-data handling

- **No imputation.** Rows with any NaN model feature or NaN `regime` are dropped:

```python
# 4_Classifier/data.py
df = df.dropna(subset=feature_cols + ["regime"])
```

- Raw NaN counts in the base columns before the drop: `weighted_norm_avg`=75,
  `weighted_deriv`=72, `norm_combined_avg`=82, `price`=11. The dominant driver of
  the 858-row drop is the longest feature window (`d2_192` needs 2×192 = 384 bars
  of warm-up ≈ 4 days), which is why modeling starts ~4 days after the raw start.

---

## 4. Feature engineering

### 4.1 Feature count and provenance

- **87 model features total.** Verified against
  `4_Classifier/model_artifacts/feature_cols.json`.
- Composition: **3 base indicators + (3 indicators × 4 windows × 7 derived) = 3 + 84 = 87.**
- `price` is **excluded** from the model (kept only for backtests/dashboards).
- All regime-defining columns are dropped as leakage (see §4.6).

### 4.2 Base indicators (3)

These are **pre-computed Prometheus recording rules**; their internal formulas are
defined upstream in Prometheus, **not in this repo**. They are all transforms of
`crypto_last_price`:

| Feature | Prometheus series | Nature |
|---------|-------------------|--------|
| `weighted_norm_avg_16h_24h_48h` | `job:crypto_last_price:weighted_normalized_avg:16h:24h:48h` | Normalized weighted average over 16h/24h/48h |
| `weighted_deriv_24h_48h_7d` | `job:crypto_last_price:weighted_deriv:24h:48h:7d` | Weighted derivative over 24h/48h/7d |
| `norm_combined_avg` | `job:crypto_last_price:normalized_combined_avg` | Normalized combined average |

Because these are **normalized** upstream, the pipeline applies **no additional
scaling/standardization**.

### 4.3 Derived momentum-state features (Stage 3)

Defined in `3_Momentum_State_Encoder/momentum_state.py`. For each base feature `f`
and window `w ∈ {16, 32, 96, 192}` (≈ 4h, 8h, 24h, 48h), 7 columns are produced:

| Column | Formula / definition |
|--------|----------------------|
| `f_d1_w` | Windowed first derivative: `f.diff(w) / w` |
| `f_d2_w` | Second derivative: `d1.diff(w) / w` |
| `f_d1_fast_w` | Fast first derivative: `f.diff(k) / k`, where `k = clamp(w // 8, 2, 24)` → `w=16→k=2`, `32→4`, `96→12`, `192→24` |
| `f_mstate_w` | Integer 9-class momentum state (0–8), see §4.4 |
| `f_mstate_duration_w` | Consecutive bars the current state has held (run length, ≥1) |
| `f_pre_cross_warning_w` | Bool: `f>0` and state ∈ {POS_DECEL, POS_FLAT} and `f < 2·threshold` |
| `f_pre_trough_warning_w` | Bool: `d1_fast < 0` and `d1_fast > -2·epsilon_fast` |

Booleans are cast to `int8` before training (`data.py`).

### 4.4 The 9 momentum states (`f_mstate_w`)

Assigned by `_classify_state` in priority order (highest wins). `thr` and `eps` are
learned dead-zone/flatness constants (§4.5):

| Code | State | Condition (causal) |
|------|-------|--------------------|
| 8 | DERIVATIVE_PEAK | `d1_fast` flipped +→− within last 3 bars |
| 7 | DERIVATIVE_TROUGH | `d1_fast` flipped −→+ within last 3 bars (top priority) |
| 6 | POSITIVE_ACCELERATING | `f>thr, d1>0, d2>0` |
| 5 | POSITIVE_DECELERATING | `f>thr, d1>0, d2<0` |
| 4 | POSITIVE_FLATTENING | `f>0, |d1|<eps` |
| 3 | CROSSING_DOWN | `f` flipped +→− within last 3 bars |
| 2 | NEGATIVE_DECELERATING | `f<-thr, d1<0, d2>0` |
| 1 | NEGATIVE_ACCELERATING | `f<-thr, d1<0, d2<0` |
| 0 | NEUTRAL | `|f|<thr` (fallback) |

### 4.5 Learned constants (fit on train fraction only)

Constants are estimated on the **first 70% of the encoder's input, chronologically**:

```python
THRESHOLD_PCTL = 30.0   # thr[f]       = 30th percentile of |f|      over first 70%
EPSILON_PCTL   = 10.0   # eps[f][w]    = 10th percentile of |d1_w|   over first 70%
                        # eps_fast[f][w]= 10th percentile of |d1_fast_w| over first 70%
TRAIN_FRACTION = 0.70
```

These frozen constants are exported to
`5_Live_Service/artifacts/encoder_artifacts.json` and reused verbatim live.

### 4.6 Normalization / differencing / lagging / winsorizing

- **Normalization/scaling:** none added (base inputs pre-normalized upstream).
- **Differencing:** yes — `d1`, `d2`, `d1_fast` are differences.
- **Lagging:** implicit via `diff`/`rolling`; no explicit lag features.
- **Winsorization / outlier clipping:** none.
- **Encoding:** momentum states are ordinal ints; warnings are 0/1.

### 4.7 Future-data check

- All derived features use only `diff`, trailing `rolling`, and backward sign-flip
  detection — **no `shift(-N)`**. Verified in `momentum_state.py`.
- **No current-price look-ahead** in features: everything is a function of past and
  current indicator values that exist at bar close.
- **Caveat (live vs. train):** `f_mstate_duration_w` (run length) is unbounded in
  training (computed over 13 months) but **truncated live** because the live
  service only fetches ~8 days (`RECOMMENDED_LOOKBACK_DAYS = 8`). This is a
  train/serve distribution difference, not future leakage. Duration features carry
  low importance (§8.4), so impact is minor.

---

## 5. Train / validation / test split

### 5.1 Method

Chronological, **no shuffling**, defined in `4_Classifier/data.py`:

```python
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC   = 0.15
# test = remaining 0.15
n_train = round(n * 0.70); n_val = round(n * 0.15); n_test = n - n_train - n_val
train = df.iloc[:n_train]; val = df.iloc[n_train:n_train+n_val]; test = df.iloc[n_train+n_val:]
```

### 5.2 Exact date ranges (n = 36,770 after warm-up drop)

| Split | Rows | Start (UTC) | End (UTC) |
|-------|------|-------------|-----------|
| Train | 25,739 | 2025-03-22 00:15:18 | 2025-12-20 21:00:18 |
| Validation | 5,516 | 2025-12-20 21:15:18 | 2026-02-16 08:00:18 |
| Test | 5,515 | 2026-02-16 08:15:18 | 2026-04-15 00:15:18 |

### 5.3 Per-split class distribution

| Split | CHOP | TRENDING_UP | TRENDING_DOWN | VOLATILE_EXPANSION |
|-------|------|-------------|---------------|--------------------|
| Train | 38.8% | 29.1% | 25.5% | 6.5% |
| Val | 36.6% (2,017) | 23.7% (1,305) | 30.9% (1,704) | 8.9% (490) |
| Test | 27.2% (1,501) | 32.4% (1,788) | 31.6% (1,745) | 8.7% (481) |

### 5.4 Walk-forward / purge / embargo

- **Walk-forward:** **No.** A single fixed split was used. (A reusable
  walk-forward splitter exists elsewhere in the repo at
  `2026/btc_direction_model/src/split/walk_forward.py` but is **not** used here.)
- **Purge / embargo:** **None.** Splits are adjacent with no gap.
- **Window/label overlap across boundaries (auditor flag):** feature windows span
  up to 192 bars and `d2` up to 384 bars; the label uses up to a 672-bar (7-day)
  window plus non-causal hysteresis. The first several hundred bars of val/test
  therefore share underlying price history with the tail of the preceding split.
  Without an embargo this is a mild optimistic bias at each boundary.

---

## 6. LightGBM configuration

From `4_Classifier/train.py` (defaults; no CLI overrides were used for the
committed model). Values **not** listed in the params dict fall back to LightGBM
defaults, noted explicitly.

| Parameter | Value | Source |
|-----------|-------|--------|
| `objective` | `multiclass` | explicit |
| `num_class` | 4 | explicit |
| `metric` | `multi_logloss` | explicit |
| `num_boost_round` (n_estimators) | 2000 (max) | explicit |
| **Actual rounds used** | **49** (`best_iteration`) | early stopping |
| `learning_rate` | 0.05 | explicit |
| `num_leaves` | 63 | explicit |
| `max_depth` | **-1 (unlimited)** | **LightGBM default** (not set) |
| `min_data_in_leaf` | 200 | explicit |
| `feature_fraction` | 0.85 | explicit |
| `bagging_fraction` | 0.85 | explicit |
| `bagging_freq` | 5 | explicit |
| `lambda_l1` | **0.0** | **LightGBM default** (not set) |
| `lambda_l2` | 0.1 | explicit |
| `early_stopping_round` | 100 (on `val` multi_logloss) | explicit |
| `seed` | 42 | explicit |
| `deterministic` | true | explicit |
| `verbose` | -1 | explicit |

### 6.1 Class weights (sample weights)

Inverse-frequency weights, normalized to mean 1, applied as per-row sample weights
to **both** train and val (`class_weights_inverse_freq` in `data.py`,
applied in `train.py`):

| Class | Weight |
|-------|--------|
| CHOP | 0.408 |
| TRENDING_UP | 0.544 |
| TRENDING_DOWN | 0.621 |
| VOLATILE_EXPANSION | 2.427 |

```python
sample_weight_train = y_train.map(cw)
sample_weight_val   = y_val.map(cw)
train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight_train)
val_data   = lgb.Dataset(X_val,   label=y_val,   weight=sample_weight_val, reference=train_data)
```

> **Auditor note:** weighting the validation set as well as train means early
> stopping optimizes a *weighted* logloss, and the emitted `prob_*` are **not
> calibrated posteriors** — they are skewed toward the rare class. Anything that
> thresholds these probabilities (Stage 6) is consuming distorted scores.

---

## 7. Training process

| Question | Answer |
|----------|--------|
| Hyperparameter tuning? | **No.** Fixed hand-chosen defaults in `train.py`. |
| Search method | **None** (no grid/random/Optuna/Bayesian search in the training code). |
| Metric optimized | `multi_logloss` on the validation set (for early stopping). |
| Early stopping | **Yes**, 100 rounds patience on `val` multi_logloss → stopped at iteration 49. |
| Trained once or across folds? | **Once.** No cross-validation, no per-fold retraining. |
| Determinism | `seed=42`, `deterministic=true` → reproducible. |
| Best val score | multi_logloss = 0.80441 (weighted) at iteration 49. |

Artifacts written: `model.txt`, `feature_cols.json`, `class_weights.json`,
`metrics.json`, `predictions_val.parquet`, `predictions_test.parquet`,
`train_log.txt`.

---

## 8. Performance results

Classification only (no probability calibration was performed). Metrics below were
recomputed from the saved prediction parquets.

### 8.1 Headline

| Split | Accuracy | Macro-F1 | Log-loss |
|-------|----------|----------|----------|
| Train | not persisted (per-row train predictions were not saved) | — | — |
| Validation | **0.7331** | **0.7095** | **0.6799** |
| Test | **0.6945** | **0.6379** | **0.7314** |

Baselines for context (test split): majority-class (CHOP) ≈ 27.2%; a persistence
baseline was **not** computed. ROC-AUC / PR-AUC were **not** computed by the
pipeline.

### 8.2 Per-class report — VALIDATION

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| CHOP | 0.679 | 0.781 | 0.726 | 2,017 |
| TRENDING_UP | 0.768 | 0.759 | 0.764 | 1,305 |
| TRENDING_DOWN | 0.798 | 0.717 | 0.755 | 1,704 |
| VOLATILE_EXPANSION | 0.684 | 0.522 | 0.593 | 490 |

Confusion matrix (rows = true, cols = predicted; order CHOP, UP, DOWN, VOL):

```
             CHOP    UP  DOWN   VOL
CHOP        1575   237   174    31
UP           265   991     8    41
DOWN         430     6  1222    46
VOL           51    56   127   256
```

### 8.3 Per-class report — TEST

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| CHOP | 0.543 | 0.808 | 0.650 | 1,501 |
| TRENDING_UP | 0.838 | 0.745 | 0.789 | 1,788 |
| TRENDING_DOWN | 0.817 | 0.648 | 0.723 | 1,745 |
| VOLATILE_EXPANSION | 0.500 | 0.320 | 0.390 | 481 |

Confusion matrix (rows = true, cols = predicted):

```
             CHOP    UP  DOWN   VOL
CHOP        1213   164   114    10
UP           357  1332    13    86
DOWN         548     8  1131    58
VOL          114    86   127   154
```

**Observations for auditors:**
- VOLATILE_EXPANSION recall collapses from 0.522 (val) → 0.320 (test); it is the
  weakest class despite the 2.4× up-weighting.
- CHOP precision drops 0.679 → 0.543 test: the model over-predicts CHOP as its
  share shrinks (distribution shift).
- The val→test degradation across every class is consistent with regime drift +
  a single non-embargoed split.

### 8.4 Feature importance (by total gain)

Top 15 of 87 (full list in `train_log`/regenerable from `model.txt`):

| Rank | Feature | Gain share | Split count |
|------|---------|-----------|-------------|
| 1 | `weighted_deriv_24h_48h_7d` | 15.89% | 809 |
| 2 | `weighted_deriv_24h_48h_7d_d1_fast_16` | 9.52% | 461 |
| 3 | `weighted_norm_avg_16h_24h_48h` | 6.89% | 702 |
| 4 | `weighted_deriv_24h_48h_7d_d1_96` | 5.81% | 470 |
| 5 | `weighted_norm_avg_16h_24h_48h_d1_fast_16` | 5.57% | 517 |
| 6 | `weighted_deriv_24h_48h_7d_mstate_96` | 3.64% | 61 |
| 7 | `weighted_norm_avg_16h_24h_48h_d1_32` | 3.57% | 238 |
| 8 | `weighted_norm_avg_16h_24h_48h_d1_fast_32` | 3.40% | 333 |
| 9 | `weighted_deriv_24h_48h_7d_d1_32` | 3.21% | 343 |
| 10 | `norm_combined_avg` | 2.57% | 465 |
| 11 | `weighted_deriv_24h_48h_7d_d2_32` | 2.53% | 402 |
| 12 | `weighted_norm_avg_16h_24h_48h_d1_96` | 2.43% | 360 |
| 13 | `weighted_deriv_24h_48h_7d_d2_16` | 2.42% | 304 |
| 14 | `weighted_deriv_24h_48h_7d_d1_fast_32` | 2.39% | 248 |
| 15 | `weighted_norm_avg_16h_24h_48h_d2_16` | 2.23% | 289 |

Concentration and dead weight:
- The `weighted_deriv` family dominates; the derivative-momentum signals carry the
  model.
- `mstate_duration_*` features together ≈ 4.3% of gain.
- **~24 features have ≈ 0 gain**, including **every `pre_cross_warning_*` and
  `pre_trough_warning_*`** column and most `norm_combined_avg_mstate_*` columns
  (14 features have exactly 0 gain / 0 splits). These are candidates for removal.

### 8.5 Performance by time period / regime

Only the two out-of-sample periods above exist (val = late-Dec 2025→mid-Feb 2026,
test = mid-Feb→mid-Apr 2026). No per-fold or per-month breakdown was produced
because walk-forward was not used.

---

## 9. Trading / backtest performance

### 9.1 Signal generation

The model's probabilities are **not** turned into trades directly. Stage 6
(`6_Bullish_Entry_State_Machine`) consumes the `prob_*` vector and runs a
finite-state machine `NEUTRAL → CHOP_BASE → EXPANSION_ALERT →
BULLISH_CONFIRMATION → LONG_ENTRY`. Decision thresholds (`config.yaml`):

```yaml
chop_lookback_bars: 12
chop_dominance_threshold: 0.35
required_chop_dominance_ratio: 0.65
volatile_expansion_threshold: 0.25
volatile_expansion_rise_threshold: 0.15
volatile_expansion_lookback_bars: 4
trend_up_threshold: 0.40
trend_spread_threshold: 0.10
require_trend_crossing: false
range_lookback_bars: 96
breakout_buffer_pct: 0.001
max_signal_age_bars: 8
minimum_model_confidence: 0.55
minimum_confidence_gap: 0.10
entry_cooldown_bars: 96
allow_reentry_while_in_position: false
```

`ENTER_LONG` requires: bullish confirmation **and** price breakout above the prior
96-bar high ×(1+0.001) **and** top-class confidence ≥ 0.55 with a ≥ 0.10 gap.

### 9.2 Committed backtest result

From `6_Bullish_Entry_State_Machine/reports/` (the only committed backtest):

| Metric | Value |
|--------|-------|
| Bars processed | 1,230 |
| States reached | NEUTRAL (779), CHOP_BASE (388), EXPANSION_ALERT (63) |
| BULLISH_CONFIRMATION reached | **0** |
| LONG_ENTRY / `ENTER_LONG` events | **0** (`enter_long_events.json == []`) |
| Trades | **0** |

**Consequently, none of the following exist for this model:** win rate, average
win/loss, profit factor, max drawdown, Sharpe, Sortino, number of trades > 0.

### 9.3 Assumptions

- **Fees / slippage:** **not modeled** anywhere in Stage 6.
- **Signal latency:** **not modeled.** Breakout uses the current bar's price vs.
  prior-bar range; an `ENTER_LONG` is assumed actionable at the same bar.
- **Out-of-sample:** the backtest simply replays scored bars through the state
  machine; it is **not** segmented to the model's test window, and it produced no
  trades regardless.

> **Bottom line for §9:** there is **no evidence of profitability** because the
> strategy layer has never generated a trade in the committed artifacts. Any claim
> of trading edge is currently unsupported.

---

## 10. Leakage and live-readiness audit

### 10.1 Identified leakage vectors

1. **Non-causal label hysteresis (low-moderate).** `_apply_hysteresis` uses future
   bars to erase short runs (§2.4). The training label near transitions encodes a
   few bars of future information. Fix: confirm-after-k-bars (causal) hysteresis.
2. **Encoder fit boundary mismatch (low).** Stage 3 learns thresholds/epsilons on
   the first 70% of its **37,628-row** input; the classifier splits 70% of the
   **36,770-row** post-drop frame. The two 70% boundaries differ by ~hundreds of
   bars, so a thin slice of the classifier's validation window influenced the
   encoder percentiles. Fix: fit encoder constants strictly on the classifier's
   train index.
3. **No purge/embargo between splits (low-moderate).** Overlapping feature/label
   windows at split boundaries (§5.4) create mild optimistic bias.

**No hard leakage found:** the regime-defining columns are explicitly excluded —

```python
# 4_Classifier/data.py — LEAKY_COLUMNS
["regime","regime_raw","regime_name","regime_start","bars_in_regime",
 "rolling_return_4h","rolling_return_12h","rolling_return_24h","rolling_return_7d",
 "realized_volatility_4h","realized_volatility_24h"]
```

and `price` is excluded. Feature importances contain no leaky column, and val
logloss (0.68) is not suspiciously low — both consistent with no hard leak.

### 10.2 Features that may differ at prediction time

- `*_mstate_duration_*`: run length is **truncated live** (8-day fetch vs. 13-month
  training) → systematically smaller live values. Low importance, but a real
  train/serve skew.
- All features depend on the **exact same Prometheus recording rules** used in
  training. If those rules' definitions, resolution, or `step` change, live
  features silently diverge from training. There is no runtime schema/versific­ation
  check on the indicator definitions.

### 10.3 Is the validation method realistic for time-series trading?

**Partially.** Chronological split with no shuffle is correct in spirit, but:
- single split (no walk-forward) → high-variance, optimistic estimate;
- no embargo → boundary leakage;
- metrics are classification-only against a partly future-aware label; no
  economic (PnL) evaluation.

For trading use, walk-forward with embargo and a PnL-based objective is required
before these numbers can be trusted.

### 10.4 Is the model likely overfit?

**Not grossly overfit to noise, but optimistically evaluated.** Evidence:
- Early stop at iteration **49/2000** with `min_data_in_leaf=200` → a shallow,
  regularized ensemble; not a memorizer.
- However, the **val→test drop** (acc 0.733→0.694, macro-F1 0.709→0.638) shows the
  single-split estimate does not hold under regime drift. The realistic
  out-of-sample performance is the **test** figure or worse.
- 24 near-zero-importance features add noise surface without value.

### 10.5 What must change before live use

1. **Make the label causal** (confirm-after-k hysteresis) so it is verifiable live.
2. **Add walk-forward validation with a purge/embargo gap**; report per-fold.
3. **Calibrate probabilities** (or train unweighted with per-class thresholds)
   before Stage 6 consumes `prob_*`; then re-tune Stage 6 thresholds.
4. **Add baselines** (persistence + the price rule computed live) and prove the
   model beats the rule on **lead time at transitions**, not bar accuracy.
5. **Run the price rule live alongside** the model for drift monitoring.
6. **Fix the encoder fit boundary** to the train index; **prune** zero-importance
   features.
7. **Backtest the full chain with fees/slippage/latency** and confirm > 0 trades;
   the current strategy produces none, so re-tune Stage 6 or the signal mapping.
8. **Pin the indicator recording-rule definitions** and add a runtime check that
   live series match the training schema.

---

## Appendix A — Reproduction

```bash
# Retrain (deterministic, seed=42)
cd StagedBuild/4_Classifier
python3 train.py

# Regenerate the metrics in this document
python3 evaluate.py --split val
python3 evaluate.py --split test
```

## Appendix B — Source-of-truth files

| Artifact | Path |
|----------|------|
| Label logic | `2_Build_Labels/build_labels.py` |
| Feature encoder | `3_Momentum_State_Encoder/momentum_state.py` |
| Split + leakage list | `4_Classifier/data.py` |
| Trainer + hyperparameters | `4_Classifier/train.py` |
| Saved model | `4_Classifier/model_artifacts/model.txt` |
| Feature list (87) | `4_Classifier/model_artifacts/feature_cols.json` |
| Metrics | `4_Classifier/model_artifacts/metrics.json` |
| Class weights | `4_Classifier/model_artifacts/class_weights.json` |
| Val/test predictions | `4_Classifier/model_artifacts/predictions_{val,test}.parquet` |
| Strategy config | `6_Bullish_Entry_State_Machine/config.yaml` |
| Backtest output | `6_Bullish_Entry_State_Machine/reports/` |
