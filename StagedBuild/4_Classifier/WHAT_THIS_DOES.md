# Stage 4 — Regime Nowcaster

This stage is the actual machine-learning model. Stages 1–3 only
*prepared* data; Stage 4 is what looks at that data and **emits a
regime probability for the current bar**.

The output is a feature, not a decision. Downstream systems consume
the probabilities however they want — that is not this stage's job.

---

## What this stage really does (plain English)

Up until now we have:

- **Stage 1** — pulled BTC price + indicator history from Prometheus into a parquet.
- **Stage 2** — looked at the price history *in hindsight* and labeled each
  bar with one of four regimes: `CHOP`, `TRENDING_UP`, `TRENDING_DOWN`,
  `VOLATILE_EXPANSION`. This labeling uses *future-aware* logic
  (rolling returns over the last day vs the last week, etc.) — fine
  because we're just generating training labels, never decisions.
- **Stage 3** — converted every continuous indicator into a 9-class
  momentum state plus derivatives, durations, and warning flags. ~140
  numeric features per bar, every one strictly backward-looking.

Now we want a model that, given **only the Stage 3 features** at a
single bar, can predict the Stage 2 regime label. This is a supervised
classification problem:

```
input  : ~87 momentum-state features at bar t
output : a probability over {CHOP, TRENDING_UP, TRENDING_DOWN, VOLATILE_EXPANSION}
```

Why is this useful? Because the input features are **causal** (they
only use past data), the model can be deployed live: pull the latest
bar, run it through Stage 3, feed it to the model, and you get a
real-time regime probability and a confidence number. Whatever
consumes that — a downstream system, a dashboard, an alert pipeline —
gets a clean, standardized signal.

---

## The "no leakage" rule (most important thing in this stage)

Stage 2 produces the regime label using formulas like
`rolling_return_24h > 1%`. Those columns end up next to the regime
in the parquet. **If we trained the model on those columns it would
trivially get 100% accuracy** by re-deriving Stage 2's rules. That
model would be useless live and wouldn't even reveal it was useless
until you deployed it.

So `data.py` enforces a **leakage drop list**:

```python
LEAKY_COLUMNS = [
    "regime", "regime_raw", "regime_name", "regime_start", "bars_in_regime",
    "rolling_return_4h", "rolling_return_12h", "rolling_return_24h", "rolling_return_7d",
    "realized_volatility_4h", "realized_volatility_24h",
]
```

These never enter the model. The classifier only sees Stage 3
momentum-state features and the three raw indicator columns those were
derived from. **Don't relax this.**

---

## The chronological split (second most important thing)

A regular ML setup randomly shuffles rows into train/val/test. **You
cannot do that with time series.** Random shuffling lets the model
learn from a Tuesday and predict a Monday, which never happens live.

Stage 4 splits **chronologically**:

```
train : first 70%  -- 2025-03-14 -> 2025-12-18
val   : middle 15% -- 2025-12-18 -> 2026-02-15
test  : last 15%   -- 2026-02-15 -> 2026-04-15
```

The model only ever sees data from the past while training, and is
evaluated on data from a strictly later period. This is how it would
behave live.

---

## What the model actually is

A **LightGBM gradient-boosted decision tree** trained as a 4-class
softmax classifier (`objective=multiclass`).

Why LightGBM and not a neural net?

- ~90 features, ~80k training rows. That's small. Trees crush neural
  nets on small tabular data.
- No scaling needed (boolean columns, integer states, and floats all
  go in raw).
- Built-in feature importance — you immediately see which features the
  model cares about.
- Trains in ~10 seconds on a laptop.
- LightGBM has very mature support for class weights and early stopping.

We use **inverse-frequency class weights** so the rare
`VOLATILE_EXPANSION` (~3% of data) gets ~3× the weight of `CHOP`
(~60%). Without this, the model would happily ignore the rare class.
The weights are normalized so they average 1 and are saved as
`class_weights.json` for reproducibility.

We use **early stopping on val log-loss** (default: 100 rounds without
improvement). On the current dataset the best iteration was ~36 — much
fewer than the 2000-round cap — meaning the validation score stops
improving early and adding more boosting rounds just memorizes the
training set.

---

## Files

| File | Purpose |
|------|---------|
| `data.py` | Loads Stage 3 parquet, drops leakage columns, makes the chronological split, computes class weights. Used by every other script. |
| `train.py` | Trains the LightGBM classifier and saves all artifacts. |
| `evaluate.py` | Loads `train.py`'s saved predictions and prints the metrics that actually matter (confusion matrix, per-class report, transition accuracy, confidence calibration, feature importances). Also writes plots. |
| `predict.py` | Inference helper. Loads the trained model and scores any parquet that has the right Stage-3 schema. This is the class a live service would import. |
| `model_artifacts/` | Created by `train.py`. Contains `model.txt` (the saved LightGBM booster), `feature_cols.json`, `class_weights.json`, `metrics.json`, and per-row predictions on val + test. |
| `plots/` | Confusion matrices and feature-importance bar chart, written by `evaluate.py`. |

---

## How to run it

```bash
cd StagedBuild/4_Classifier

python3 train.py
python3 evaluate.py --split test
python3 evaluate.py --split val
```

Optional: tune hyperparameters from the CLI.

```bash
python3 train.py --num-leaves 127 --learning-rate 0.03 --early-stopping 200
```

If you change anything upstream (Stage 2 thresholds, Stage 3 windows),
re-run Stage 3 to regenerate `btc_data_15m_mstate.parquet` and then
re-run `train.py`.

---

## Interpreting the training output

When `train.py` runs you should see something like:

```
Dataset summary
---------------
features          : 87
train rows        : 79,006  (CHOP=60.6%, TRENDING_UP=19.4%, TRENDING_DOWN=17.9%, VOLATILE_EXPANSION=2.1%)
val   rows        : 16,930  (CHOP=59.8%, TRENDING_UP=16.8%, TRENDING_DOWN=19.7%, VOLATILE_EXPANSION=3.7%)
test  rows        : 16,929  (CHOP=48.7%, TRENDING_UP=23.7%, TRENDING_DOWN=23.1%, VOLATILE_EXPANSION=4.4%)
```

Things to look at:

- **Class shares should be similar across splits.** If `train` is 60%
  CHOP but `test` is 90% CHOP, the model is being asked to predict
  on a market it never saw. Re-check Stage 2's hysteresis or extend
  Stage 1's data range.
- **Best iteration < n_estimators.** If best iteration equals the cap,
  the model didn't converge — bump `--n-estimators` or lower the
  learning rate.
- **`val_logloss` should be lower than the "always predict CHOP"
  baseline of `−ln(0.6) ≈ 0.51` for a 60-40 dataset.** Anything around
  0.6–0.9 is sane for this problem; below ~0.4 means you probably have
  a leakage column slipping through.

---

## Interpreting the evaluation output

`evaluate.py` prints six blocks. Here's how to read each.

### 1. Headline (overall accuracy + macro F1)

```
Overall accuracy :  74.9%
Macro F1         : 0.6901
```

- **Accuracy** = % of bars where the prediction matched the true label.
- **Macro F1** averages the F1 score across the 4 classes *equally*. A
  model that gets CHOP perfect but ignores VOLATILE_EXPANSION will look
  great on accuracy and bad on macro F1. We care more about macro F1.

For comparison: always predicting CHOP would yield ~49% accuracy on
test, ~0.16 macro F1. Our model is well above both.

### 2. Per-class report

```
                    precision    recall  f1-score   support
              CHOP      0.739     0.837     0.785      8251
       TRENDING_UP      0.758     0.711     0.734      4018
     TRENDING_DOWN      0.783     0.660     0.716      3910
VOLATILE_EXPANSION      0.643     0.444     0.525       750
```

- **Precision** = "of the bars I called X, how many really were X?"
  High precision = few false alarms.
- **Recall** = "of the bars that really were X, how many did I catch?"
  High recall = few misses.
- **F1** = harmonic mean of the two; rewards being good at both.

What we want: high precision on the trending classes (don't fire false
alarms). The model has 76–78% precision on TRENDING_UP and
TRENDING_DOWN — solid. Recall is lower (66–71%) because the model
prefers to "stay neutral" (predict CHOP) when uncertain. **For a
detection system, that's the safer failure mode** — abstaining on a
real trend is recoverable, mis-labelling a downtrend as an uptrend is
not.

### 3. Confusion matrix

```
true\pred               CHOP  TRENDING_U  TRENDING_D  VOLATILE_E
CHOP                   83.7%        8.5%        7.2%        0.6%
TRENDING_UP            27.5%       71.1%        0.2%        1.2%
TRENDING_DOWN          30.8%        0.9%       66.0%        2.3%
VOLATILE_EXPANSION     17.3%       23.6%       14.7%       44.4%
```

The killer numbers are the **off-diagonal cells where direction
matters**:

- TRENDING_UP misclassified as TRENDING_DOWN: **0.2%** ✓
- TRENDING_DOWN misclassified as TRENDING_UP: **0.9%** ✓

The model almost never gets the *direction* wrong. When it errors, it
errors toward CHOP (i.e. it abstains).

### 4. Confidence vs accuracy

```
bucket                 n   share   accuracy
[0.6, 0.7)         3,305   19.5%      70.8%
[0.7, 0.8)         4,229   25.0%      83.9%
[0.8, 0.9)         3,872   22.9%      92.0%
[0.9, 1.001)         212    1.3%     100.0%
```

The model's confidence is **well-calibrated**: the higher the model's
predicted probability, the more often it is right. Downstream consumers
can use confidence as a quality filter — a `confidence > 0.8` filter
gives you a 92%-accurate signal that fires on roughly 24% of bars.

### 5. Transition accuracy

```
Transition accuracy (tolerance: +/- 4 bars):
  Transitions in split    : 524
  Transitions caught (±4): 355  ( 67.7%)
```

When the *true* regime flips (say, CHOP → TRENDING_UP), did the model
predict the new regime within ±1 hour (4 bars at 15-min)? **67.7% of
the time, yes.** That's the metric that tells you whether the model
will catch real moves vs miss them.

If transition accuracy is high but per-class recall is low, the model
is fast but misses sustained regimes. If it's the opposite, the model
is slow to react. We have both reasonably balanced.

### 6. Feature importance

```
Top features by total gain:
  norm_combined_avg                                               71,628
  weighted_deriv_24h_48h_7d_d1_fast_16                            60,158
  weighted_deriv_24h_48h_7d_d1_16                                 19,983
  ...
```

`gain` is how much each feature reduced the model's loss across all
splits it appeared in. Higher = more useful.

What we **want** to see (and do):

- The `_d1_fast_*` columns (the fast inflection derivatives from
  Stage 3) are heavily used. The fast-inflection rework was the right
  call.
- Multiple windows show up in the top 25 (16, 32, 96, 192). The model
  is genuinely combining short- and long-term context.
- `mstate_duration` columns appear — the model uses how *long* a state
  has held, not just the current state.

What we **do not want** to see (and don't):

- Any column from `LEAKY_COLUMNS` in this list. If `regime` or
  `rolling_return_24h` shows up, something broke; check `data.py`.

---

## What "good" looks like

For this problem, on this data, here's a sane checklist:

| Metric | OK | Good | Great |
|--------|----|------|-------|
| Test accuracy | > 60% | > 70% | > 80% |
| Macro F1 | > 0.5 | > 0.65 | > 0.75 |
| Direction-flip error (`TRENDING_UP` <-> `TRENDING_DOWN`) | < 5% | < 2% | < 1% |
| Transition accuracy ±4 bars | > 50% | > 65% | > 80% |
| Confidence > 0.8 accuracy | > 80% | > 90% | > 95% |
| `VOLATILE_EXPANSION` recall | > 0.3 | > 0.5 | > 0.7 |

We're currently in **good** for all metrics, **great** on direction
discipline. `VOLATILE_EXPANSION` is the hardest class — it's rare and
varied — and there's clear headroom to improve it later.

---

## What this stage outputs

After running `train.py`, the artifacts in `model_artifacts/` describe
the full output contract:

| Field | Type | Meaning |
|-------|------|---------|
| `prob_CHOP` | float in [0, 1] | Probability the current bar is in the CHOP regime. |
| `prob_TRENDING_UP` | float in [0, 1] | Probability the current bar is in TRENDING_UP. |
| `prob_TRENDING_DOWN` | float in [0, 1] | Probability the current bar is in TRENDING_DOWN. |
| `prob_VOLATILE_EXPANSION` | float in [0, 1] | Probability the current bar is in VOLATILE_EXPANSION. |
| `pred_int` | int 0–3 | argmax of the four probabilities. |
| `pred_name` | str | Human-readable name of `pred_int`. |
| `confidence` | float in [0, 1] | The max of the four probabilities. |

The four probabilities always sum to 1. Whatever consumes this output
gets the *full distribution*, not just the top-1 prediction, so it can
implement its own thresholds, abstain rules, or fusion logic without
having to retrain anything.

---

## What this stage does NOT do

- **It does not look ahead.** The model never sees future bars. The
  causal-integrity test from Stage 3 plus the leakage drop list here
  enforces that.
- **It does not act on the predictions.** This is a detector. The
  output is a probability vector and a confidence — period. Whatever
  reads the output decides what to do with it.
- **It is not the only model.** The same Stage 3 output can feed an
  LSTM, a Transformer, a logistic regression, or whatever else you
  want to compare against. LightGBM is the strong baseline; better
  models can be plugged in by writing a new `train.py`.

---

## How a live service consumes this

The `RegimeClassifier` class in `predict.py` is the live entry point.
A real-time loop looks like this:

```python
from predict import RegimeClassifier

clf = RegimeClassifier()  # loads model + feature list once

# Every 15 minutes:
df = pull_latest_bars_from_prometheus(window="48h")  # need >= max(windows)
df = MomentumStateEncoder(...).fit_transform(df)     # Stage 3
predictions = clf.predict(df.tail(1))

regime     = predictions["pred_name"].iloc[0]    # e.g. "TRENDING_UP"
confidence = predictions["confidence"].iloc[0]   # e.g. 0.83
prob_up    = predictions["prob_TRENDING_UP"].iloc[0]
prob_down  = predictions["prob_TRENDING_DOWN"].iloc[0]

publish({                    # emit to whatever the consumers subscribe to
    "regime": regime,
    "confidence": confidence,
    "prob_chop":               predictions["prob_CHOP"].iloc[0],
    "prob_trending_up":        prob_up,
    "prob_trending_down":      prob_down,
    "prob_volatile_expansion": predictions["prob_VOLATILE_EXPANSION"].iloc[0],
})
```

That's the whole job of this stage — turn the latest bar into a
labelled distribution and publish it. Anything more complicated
belongs to the consumer, not the detector.
