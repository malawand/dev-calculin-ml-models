# Stage 3 — Momentum State Encoder

This stage is the **most important feature-engineering step in the entire project.**
Everything downstream (classifier training, live inference, backtest) consumes
whatever comes out of here. If the momentum state is wrong, nothing downstream
can compensate.

---

## The core insight: watch the **derivative**, not the value

Each indicator is a continuous number. A model could learn from the raw
number directly, but the most valuable trading signal isn't *"the indicator
is positive"* — that's a regime that already exists. The valuable signal is
**the moment the indicator's first derivative flips from negative to positive**.

That moment is the **trough**. It is where the indicator stopped getting
worse and started getting better. By the time the indicator itself climbs
back through zero, the move is already underway and most of the entry edge
has evaporated.

This encoder is built around capturing that moment first, and then layering
the steady-state context (above/below threshold, accelerating/decelerating)
on top.

---

## The 9 states

```
8 = DERIVATIVE_PEAK         f' just flipped pos -> neg     <-- primary sell event
7 = DERIVATIVE_TROUGH       f' just flipped neg -> pos     <-- primary buy event
6 = POSITIVE_ACCELERATING   f > thr, f' > 0, f'' > 0
5 = POSITIVE_DECELERATING   f > thr, f' > 0, f'' < 0       <-- "rally fading"
4 = POSITIVE_FLATTENING     f > 0, |f'| < epsilon          <-- pre-cross zone
3 = CROSSING_DOWN           f flipped pos -> neg           <-- lagging confirm
2 = NEGATIVE_DECELERATING   f < -thr, f' < 0, f'' > 0      <-- pre-trough zone
1 = NEGATIVE_ACCELERATING   f < -thr, f' < 0, f'' < 0
0 = NEUTRAL                 |f| < thr (fallback)
```

### Priority (highest wins, evaluated last so it overwrites earlier writes)

1. **`DERIVATIVE_TROUGH` (7)** — leading buy signal. Top priority.
2. **`DERIVATIVE_PEAK` (8)** — leading sell signal. Top priority.
3. `CROSSING_DOWN` (3) — late confirm; useful but you'd already know.
4. `POSITIVE_ACCELERATING` > `POSITIVE_DECELERATING` > `POSITIVE_FLATTENING`.
5. `NEGATIVE_ACCELERATING` > `NEGATIVE_DECELERATING`.
6. `NEUTRAL` (0) — fallback.

### Why are TROUGH/PEAK above CROSSING_DOWN?

Imagine an indicator does this over four bars: `+1.0  +0.3  -0.1  -0.4`.

- The d1 inflection (`d1` flipped from positive to negative) happened
  between bars 1 and 2.
- The actual sign cross of `f` happened between bars 2 and 3.

Both DERIVATIVE_PEAK and CROSSING_DOWN qualify on bar 3, but the peak fired
**one bar earlier**. We label bar 3 with the earlier (more informative)
event.

---

## Derivatives — windowed AND fast (the key design decision)

The encoder computes **two** first derivatives for every (feature, window):

```
d1[t]      = (f[t] - f[t-W]) / W        # full-window slope
d1_fast[t] = (f[t] - f[t-K]) / K        # fast slope, K = max(2, w//8)
d2[t]      = (d1[t] - d1[t-W]) / W      # second derivative on the windowed d1
```

`d1` is the smooth, low-noise long-window slope. `d1_fast` is the
short-window slope that responds quickly. They are used differently:

- **Steady-state classifications** (`POSITIVE_ACCELERATING`,
  `POSITIVE_DECELERATING`, `POSITIVE_FLATTENING`, `NEGATIVE_*`) use the
  **windowed** `d1` and `d2`. These benefit from heavy smoothing.
- **Inflection events** (`DERIVATIVE_TROUGH`, `DERIVATIVE_PEAK`) and the
  **`pre_trough_warning`** use **`d1_fast`**. This is what makes the
  events fire **near the actual local minimum/maximum of f** instead of
  ~W/2 bars later.

Why two derivatives? A windowed slope `(f[t] - f[t-W])/W` only crosses
zero ~W/2 bars *after* f hit its local extremum. For w=192 that's a
24-hour lag — useless as a "buy here" signal. The fast slope drops the
detection lag by 5-20× while the regime states keep their long-window
context.

```
inflection_window K (≈ w/8, clamped to [2, 24]):
    w=16  -> K=2     median trough lag  9 bars -> 2 bars
    w=32  -> K=4     median trough lag 18 bars -> 3 bars
    w=96  -> K=12    median trough lag 50 bars -> 6 bars
    w=192 -> K=24    median trough lag 99 bars -> 10 bars
```

Everything is strictly backward-looking. `diff()` only ever reads the
past; this is enforced by an explicit causal-integrity test in the
test suite.

At 15-minute bars, the default regime windows translate to:

| Bars | Human window |
|------|--------------|
| 16   | 4 hours      |
| 32   | 8 hours      |
| 96   | 24 hours     |
| 192  | 48 hours     |

The encoder runs the full state machine independently per regime
window, so the model sees short-term and long-term context as
separate features.

---

## Per-feature thresholds and epsilons

A single global `threshold = 0.01` would be useless — different indicators
live on totally different scales. Instead the encoder *learns* them from the
training portion of the data:

- `threshold[feat]` = **30th percentile of `|feat|`** on the first 70% of
  the data (chronological). This is the "too small to mean anything"
  dead-zone for f.
- `epsilon[feat][window]` = **10th percentile of `|d1|`** on the first
  70%. This is the "basically flat" tolerance for the windowed
  derivative; it adapts per window because longer-window derivatives
  have smaller magnitudes.
- `epsilon_fast[feat][window]` = **10th percentile of `|d1_fast|`** on
  the first 70%. Powers the `pre_trough_warning` (an imminent zero-cross
  on the fast derivative).

Only the first 70% of the data is used to fit thresholds, so nothing from
the validation / test tail leaks into the dead-zone definition.

---

## Output columns per (feature, window) pair

| Column | Description |
|--------|-------------|
| `{feat}_d1_{w}` | Windowed first derivative (change per bar averaged over W bars). Smooth; powers the steady-state classifications. |
| `{feat}_d2_{w}` | Windowed second derivative (change of `d1` over W bars). Tells acceleration vs deceleration. |
| `{feat}_d1_fast_{w}` | **Fast first derivative** computed over `K = max(2, w//8)` bars. Drives `DERIVATIVE_TROUGH`, `DERIVATIVE_PEAK`, and `pre_trough_warning`. Responds quickly so events fire near the actual local extremum. |
| `{feat}_mstate_{w}` | The 9-class state integer (0–8). |
| `{feat}_mstate_duration_{w}` | Consecutive bars in the current state. Long durations are stronger context (an established `POSITIVE_DECELERATING` is more meaningful than a fresh one). |
| `{feat}_pre_cross_warning_{w}` | Boolean. Fires when state ∈ {`POSITIVE_DECELERATING`, `POSITIVE_FLATTENING`}, `f > 0`, and `f < 2·threshold`. The lead-in to a downward sign cross of f. |
| `{feat}_pre_trough_warning_{w}` | Boolean. Fires when `d1_fast < 0` and `d1_fast > -2·epsilon_fast`. **The lead-in to `DERIVATIVE_TROUGH` — the buy signal one to several bars before it actually triggers.** Fully decoupled from any state — only depends on the fast derivative being small and negative. |

The two warning columns are designed to be the most actionable features
for a downstream classifier: they fire *before* the headline event happens.

---

## Why we kept CROSSING_DOWN at all

A late confirm is still useful. Some downstream signals only matter once
the actual sign cross of `f` has been validated. `CROSSING_DOWN` lets the
model condition on "the regime really did flip", separate from "the
derivative inflected". It just isn't the primary trigger.

---

## Causal integrity

Every column is strictly backward-looking. The test suite includes a
specific test that mutates *future* bars in the input and asserts that
*past* outputs are unchanged. This is the lookahead-leakage guardrail
that keeps the whole downstream model honest.

---

## Files

| File | Purpose |
|------|---------|
| `momentum_state.py` | `MomentumStateEncoder` class, `_classify_state`, `_crossing_mask`, `_pre_cross_warning`, `_pre_trough_warning`, `quick_visualize`, CLI entry point. |
| `test_momentum_state.py` | 20 tests: 9-state coverage, derivative-inflection priority, causality, duration counter, per-feature thresholds, both warnings, NaN behaviour, lag-reduction. |
| `visualize_states.py` | Generates PNG diagnostic plots into `plots/` from `btc_data_15m_mstate.parquet`. Includes a state-distribution stacked bar, four-panel diagnostics, and BTC-price-with-state-ribbon plots. |
| `verify_inflections.py` | Independent verifier that re-derives every TROUGH/PEAK event from scratch and compares against the encoder. Two checks (mechanical + local-extremum) plus 3-panel overlay plots. **This is the "is the encoder really detecting troughs?" tool.** |
| `btc_data_15m_mstate.parquet` | Output produced by `python3 momentum_state.py`. Same shape as the input labeled parquet plus 21 new encoder columns per (feature × window). |
| `plots/` | PNGs produced by `visualize_states.py` and `verify_inflections.py` — sanity-check and verification images. |

---

## How to use this

Programmatic:

```python
import pandas as pd
from momentum_state import MomentumStateEncoder

df = pd.read_parquet("../2_Build_Labels/btc_data_15m_labeled.parquet")
enc = MomentumStateEncoder(
    feature_cols=["weighted_norm_avg_16h_24h_48h",
                  "weighted_deriv_24h_48h_7d",
                  "norm_combined_avg"],
    windows=[16, 32, 96, 192],
)
out = enc.fit_transform(df)
```

CLI (defaults already point at stage 2's output):

```bash
python3 momentum_state.py
```

Visual sanity check on any single feature/window pair (vertical lines mark
both warning types — crimson dashed = pre_cross, gold dotted = pre_trough):

```python
from momentum_state import quick_visualize
quick_visualize(out, "norm_combined_avg", window=96, save_path="mstate_check.png")
```

---

## Tests

```bash
python3 -m pytest test_momentum_state.py -v
```

All 20 tests must pass before trusting the output downstream.

What you should see:

```
collected 20 items
test_momentum_state.py::TestStateCoverage::test_positive_accelerating PASSED
test_momentum_state.py::TestStateCoverage::test_positive_decelerating PASSED
... (one PASSED line per test) ...
======== 20 passed in 0.37s ========
```

Each named test is doing a specific thing:

| Test class | What it verifies |
|------------|------------------|
| `TestStateCoverage` | Every one of the 9 states is reachable. We hand-build a synthetic input that should land in that exact state and assert the encoder agrees. The `test_state_names_cover_full_vocabulary` confirms the integer-to-name dictionary covers every code. |
| `TestPriority` | The priority resolution is correct. Specifically: `CROSSING_DOWN` beats `POSITIVE_DECELERATING` when both apply, and `DERIVATIVE_TROUGH` / `DERIVATIVE_PEAK` beat both of those (because the d1 flip is the earlier and more actionable event). |
| `TestCausalIntegrity` | The lookahead-leakage guardrail. Mutating data in the future half of the input must not change any past output. If this fails, the whole pipeline downstream becomes invalid. |
| `TestDurationCounter` | The `_mstate_duration_*` columns count consecutive bars correctly: 1, 2, 3, ... until the state changes, then reset to 1. |
| `TestPerFeatureThresholds` | Two features with very different scales get different learned thresholds. (We don't use a single global cutoff.) |
| `TestPreCrossWarning` | The `pre_cross_warning_*` column fires only when state ∈ {`POSITIVE_DECELERATING`, `POSITIVE_FLATTENING`}, `f > 0`, and `f < 2·threshold`. Pushing the value far above zero clears the warning. |
| `TestPreTroughWarning` | Symmetric to the above, on the buy side. `pre_trough_warning_*` fires when the FAST derivative satisfies `d1_fast < 0` and `d1_fast > -2·epsilon_fast`. |
| `TestInflectionLagReduction` | Confirms the whole reason for the fast-inflection design: TROUGH events fire close to actual local minima of f (median lag ≪ W/2, on the order of K). |
| `TestNaN` | State / duration / warning columns never contain NaN; derivative columns are NaN only for the unavoidable warm-up bars. |

If everything is `PASSED` the encoder is behaving as specified and the
output is safe to feed into the downstream classifier.

---

## Interpreting the encoder's CLI output

When you run `python3 momentum_state.py`, it prints three blocks. Here's
what each one means.

### 1. Learned thresholds

```
Learned thresholds (30th pct of |f| on first 70%):
  weighted_norm_avg_16h_24h_48h           : 0.67851
  weighted_deriv_24h_48h_7d               : 0.00357
  norm_combined_avg                       : 0.66270
```

This is the per-feature dead-zone cutoff. The interpretation is:
"30% of the time during training, this feature was within ±0.67851 of
zero, and we treat values smaller than that as **noise**."

Things to notice:
- `weighted_deriv_24h_48h_7d` is a tiny number (0.00357) because that
  feature naturally lives near zero — it's a derivative metric.
- `weighted_norm_avg_16h_24h_48h` and `norm_combined_avg` are roughly
  the same size (0.66 and 0.68) — both are normalized averages on
  similar scales.

If two features got identical thresholds it would mean they're on the
same scale; getting wildly different ones is exactly what we want.

### 2. Learned epsilons per (feature, window)

```
Learned epsilons per (feature, window):
  weighted_norm_avg_16h_24h_48h    w=  16: 0.001116
  weighted_norm_avg_16h_24h_48h    w=  32: 0.001094
  weighted_norm_avg_16h_24h_48h    w=  96: 0.000980
  weighted_norm_avg_16h_24h_48h    w= 192: 0.000849
  ...
```

Epsilon is the "is the derivative effectively zero?" tolerance. The
interpretation is: "10% of the time during training, the first
derivative over an N-bar window was smaller in magnitude than this
number — those bars are effectively flat."

Things to notice:
- Epsilon shrinks as the window gets longer. That makes sense: a
  derivative computed over 192 bars naturally has a smaller magnitude
  than one over 16 bars (you're dividing by a bigger denominator).
- `weighted_deriv_24h_48h_7d` has tiny epsilons (≈ 8e-6) because the
  feature itself is tiny — the encoder adapts.
- `norm_combined_avg` has the largest epsilons (0.001 → 0.01), which
  reflects that this feature changes faster.

If you ever see an epsilon of 0.0 for a real feature, the training
slice was too short for that window — increase the input length or
shrink the window.

### 3. State distribution per (feature, window)

```
norm_combined_avg_mstate_96: NEUTRAL=33.9%, NEGATIVE_ACCELERATING=22.4%,
NEGATIVE_DECELERATING=4.9%, CROSSING_DOWN=2.4%, POSITIVE_FLATTENING=2.1%,
POSITIVE_DECELERATING=5.3%, POSITIVE_ACCELERATING=23.0%,
DERIVATIVE_TROUGH=3.0%, DERIVATIVE_PEAK=2.9%
```

This is the share of bars that landed in each of the 9 states for a
given (feature, window) pair. How to read it:

- **`NEUTRAL` is your "uninteresting" bucket.** Healthy ranges are
  roughly 35–65% depending on how busy the indicator is. Above ~80%
  means the threshold is too aggressive (most movements are getting
  filtered out as noise). Below ~25% means the threshold is too loose.
- **`POSITIVE_ACCELERATING` and `NEGATIVE_ACCELERATING` should be roughly
  symmetric over a long enough window.** BTC trends both ways, so
  asymmetry hints at a feature that's biased one direction. Above we
  see 23.0% / 22.4% — beautifully symmetric.
- **`DERIVATIVE_TROUGH` and `DERIVATIVE_PEAK` should also be roughly
  symmetric.** Inflection events come in matched pairs: every peak
  eventually has a trough. Above we see 3.0% / 2.9% — also symmetric.
- **`CROSSING_DOWN` is only the *downward* sign flip on f itself**,
  which is why it's typically smaller than the d1-inflection events.
  It's the late confirm — by the time it fires, the d1 inflection
  already happened.
- **The faster the feature, the more inflection events.** Compare
  `norm_combined_avg` (3% trough events) vs `weighted_norm_avg_16h_24h_48h`
  (~1.2%) — the first is more reactive, generating more inflections.

A good sanity check: TROUGH%, PEAK%, and CROSSING% summed shouldn't
dwarf the steady-state classes. They're event states; if any one of
them exceeds ~10% you probably have noisy data, not real signal.

---

## Visualizations

To actually see what the encoder produced on the real data:

```bash
python3 visualize_states.py
```

This writes a set of PNG files into `plots/`:

| File | What it shows |
|------|---------------|
| `state_distribution.png` | Stacked horizontal bar chart of state shares for every (feature, window) pair. Use this to spot symmetry / bias issues at a glance. |
| `<feat>_w96_full.png` | Four-panel diagnostic for one feature at the 24-hour window: raw value, first derivative, second derivative, and the colored momentum-state strip. Crimson dashed lines mark `pre_cross_warning`; gold dotted lines mark `pre_trough_warning`. |
| `<feat>_w96_ribbon.png` | BTC price on top with a thin colored ribbon below encoding the state. This is the easiest plot for "did the encoder fire correctly during this episode?" inspection. |
| `<feat>_w96_<date_range>.png` | Same plots but zoomed to a user-selected date range. Default range is **2025-03-18 → 2025-03-20**, which covers the pump-and-dump scenario the regime detector was tuned against. |

Pick a different zoom window with:

```bash
python3 visualize_states.py --slice 2025-12-01 2025-12-15
```

### What to look for in the ribbon plots

In a healthy encoder run, when you compare BTC price (top) to the
colored ribbon (bottom):

- **Sustained uptrends** → mostly dark green (`POSITIVE_ACCELERATING`)
  with light green stretches (`POSITIVE_DECELERATING`) at the tail.
- **Sustained downtrends** → mostly dark red (`NEGATIVE_ACCELERATING`)
  with light red at the tail.
- **Tops** → orange `DERIVATIVE_PEAK` flickers right at the moment
  price stops rising and reverses, often before the value of f itself
  has rolled over.
- **Bottoms** → gold `DERIVATIVE_TROUGH` flickers right at the moment
  price stops falling and reverses, before f has climbed back.
- **Choppy / consolidation** → mostly light gray (`NEUTRAL`).
- **Black `CROSSING_DOWN` flecks** are mostly later-confirms; you
  should typically see an orange `DERIVATIVE_PEAK` shortly before
  every black mark. If you see a lot of black with no preceding orange,
  the threshold may be too small relative to epsilon.

Use the zoom plots on a few different memorable BTC events (e.g.
sharp pump days, weekend doldrums) to convince yourself the encoder
is labeling the way a human trader would.

---

## Verifying that TROUGH / PEAK events are real (run this!)

Synthetic unit tests prove the *math* of the encoder. They cannot prove
the encoder is actually firing at the *real* turning points of a real
indicator. For that, run `verify_inflections.py` against the production
parquet:

```bash
python3 verify_inflections.py
```

It performs two independent checks on every TROUGH / PEAK event the
encoder produced and prints a summary table. It also writes 3-panel
overlay plots into `plots/verify_*.png` so you can eyeball things.

### Why this script exists

The whole reason for the windowed/fast split (the "fast inflection
window" design) is so that `DERIVATIVE_TROUGH` events fire at the
**actual local minima** of the feature, not ~W/2 bars later. The unit
tests check this on toy sinusoids; this script checks it on the real
parquet.

### Output 1 — Mechanical check

```
==============================================================================
MECHANICAL CHECK
(Did d1_fast actually sign-flip within last 3 bars at every event?)
This MUST be 100% — anything less is a bug in the encoder.
==============================================================================
feature                        win  trough_n  trough_d1_ok  peak_n  peak_d1_ok
-----------------------------  ---  --------  ------------  ------  ----------
weighted_norm_avg_16h_24h_48h  16   3,446      100.00%      3,055   100.00%
weighted_norm_avg_16h_24h_48h  96   1,542      100.00%      1,520   100.00%
weighted_deriv_24h_48h_7d      16   3,152      100.00%      2,812   100.00%
weighted_deriv_24h_48h_7d      96   1,396      100.00%      1,373   100.00%
norm_combined_avg              16   15,564     100.00%      13,860  100.00%
norm_combined_avg              96   6,918      100.00%      6,622   100.00%
```

This re-derives the d1_fast sign-flip condition from scratch and asks:
"of all the bars the encoder labeled `DERIVATIVE_TROUGH`, what
percentage really did have a `d1_fast` flip from negative to positive
in the last 3 bars?". This number **must be 100%**. Anything below
100% means the state machine wrote the wrong label, which is a real
bug. **Mechanical check is the floor — not the ceiling — of correctness.**

### Output 2 — Local extremum check

```
==============================================================================
LOCAL EXTREMUM CHECK (band scaled by inflection window K)
Strict: a true local minimum/maximum exists in [t - 2K, t + K/2].
Near:   feature value at event is within 15% of band's [min, max] span.
==============================================================================
feature                        win  trough_n  trough_strict  trough_near  peak_n  peak_strict  peak_near
-----------------------------  ---  --------  -------------  -----------  ------  -----------  ---------
weighted_norm_avg_16h_24h_48h  16   3,446     100.00%          6.07%      3,055    99.97%        6.91%
weighted_norm_avg_16h_24h_48h  96   1,542     100.00%         18.35%      1,520    99.87%       16.58%
weighted_deriv_24h_48h_7d      16   3,152      99.97%          6.44%      2,812   100.00%        5.48%
weighted_deriv_24h_48h_7d      96   1,396      99.86%         15.19%      1,373    99.85%       15.15%
norm_combined_avg              16   15,564     99.98%          8.10%      13,860   99.99%        7.73%
norm_combined_avg              96   6,918      99.86%          4.78%      6,622    99.92%        5.16%
```

This is the more interesting check — it asks whether the events fire
**near actual turning points of the feature**.

- **`trough_strict` / `peak_strict`** — for each TROUGH event at bar
  `t`, look in `[t - 2K, t + K/2]`. Is there a *strict* local minimum
  in that band (a bar where `f` is lower than both neighbors)? If yes,
  the encoder found a real local minimum. We want this **as close to
  100% as possible**.
- **`trough_near` / `peak_near`** — at the event bar `t` itself, is
  `f[t]` within 15% of the band's min/max range? This is the harder
  test: not just "a trough exists nearby" but "the encoder fired
  *at* the trough". Lower numbers here are expected (the encoder
  fires K bars *after* the trough by design — a few bars of climb has
  already happened).

Healthy signal:
- `*_strict` ≥ 99% across all features and windows. ✓
- `*_near` is feature-dependent. Smooth features score higher; noisy
  features score lower because they have many small wiggles inside the
  band, so the value at `t` is rarely the band's true min.

If `*_strict` ever drops below ~95%, the inflection window K is too
short for that feature and is firing on noise rather than real turning
points.

### Output 3 — Detection lag (printed by the helper script)

The lag analysis (`p25 / median / p75` bars between the last real local
minimum and the TROUGH event) is what proves the design works:

| Window | Old (`W/2`) | New median lag | Speedup |
|--------|-------------|----------------|---------|
| w=16   | 8 bars      | **2 bars**     | 4.5×    |
| w=32   | 16 bars     | **3 bars**     | 6×      |
| w=96   | 48 bars     | **5–6 bars**   | 9×      |
| w=192  | 96 bars     | **5–10 bars**  | 10–20×  |

Before the fast-inflection design, a `w=192` (48-hour) trough event
fired 24 hours after the actual minimum. Now it fires within 2.5
hours — actually usable as a buy signal.

### Output 4 — Visual overlays

`verify_inflections.py` also writes a set of PNGs:

```
plots/verify_<feature>_w96_full.png
plots/verify_<feature>_w16_2025-03-18_to_2025-03-20.png
plots/verify_<feature>_w96_2025-03-18_to_2025-03-20.png
```

Each plot has 3 stacked panels:

| Panel | What it shows |
|-------|---------------|
| **Top** | Raw feature value over time. Gold dots = `DERIVATIVE_TROUGH` events. Orange dots = `DERIVATIVE_PEAK`. **In a healthy encoder, gold dots sit at the bottoms of the wave and orange dots sit at the tops.** This is the picture you want. |
| **Middle** | `d1_fast` (the fast derivative) with the same gold/orange dots overlaid. **Dots sit exactly on the zero line** — that's the proof that events fire on `d1_fast` zero-crossings. |
| **Bottom** | `d1` (the windowed derivative) with the dots overlaid. Dots are *not* at zero here, because events no longer fire on `d1`; this panel just shows what the windowed slope looked like at the moment of the event for context. |

Open one or two of these and confirm the gold dots sit at the dips of
the top-panel wave. If they do, the encoder is doing its job.

Pick a different zoom window with:

```bash
python3 verify_inflections.py --slice 2025-12-01 2025-12-15
```

### Quick recipe — full sanity pass

```bash
# 1. Regenerate the encoded parquet from the labeled stage 2 output.
python3 momentum_state.py

# 2. Confirm the encoder math is right on toy data.
python3 -m pytest test_momentum_state.py -v

# 3. Confirm the encoder is right on real data.
python3 verify_inflections.py

# 4. Eyeball the overlay plots.
open plots/verify_*_2025-03-18_to_2025-03-20.png
```

If steps 2 and 3 both pass clean and the overlay plots show gold dots
sitting at the bottoms of the wave, the encoder is ready to feed the
downstream classifier.
