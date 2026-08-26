# Frozen timestamp rule (2026-08-24)

The live market-motion vector uses **confirmation time**, not the hindsight peak/trough time.

```text
state_effective_time = feature_available_time
```

Do not use hindsight-at-the-extremum in training and confirmation-time in testing. Training, validation, testing, live inference, and historical analogue matching must all use this same point-in-time rule.

## Example

```text
2:00 — the curve reaches what will later be identified as a peak
3:00 — not enough evidence to confirm it
4:00 — the peak is confirmed
```

At 2:00 and 3:00 the previously known state is kept (null until the first confirmation). At 4:00, once confirmation is available, the state changes. The 2:00 and 3:00 feature rows are never rewritten.

If the 4:00 close is required to confirm the turn, the state is available only after that close plus processing delay. A simulated trade cannot enter at the same close used to confirm.

## Clocks

These are separate. An hourly label is not, by itself, the time the feature became usable.

| Clock | Meaning |
|---|---|
| `observation_time` | Bar whose close is used as the feature (`observation_time_utc`) |
| `feature_available_time` | `observation_time` + processing delay (`feature_available_lag_hours`, default 1h) |
| `decision_time` | `>= feature_available_time` (`+ decision_latency_hours`, default 0) |
| `outcome_entry_time` | First executable price after `decision_time` (default: same as `decision_time`) |

If the state needs the close of the bar ending at 4:00:

```text
observation_time       = 4:00
feature_available_time = 4:00 + processing delay
decision_time          >= feature_available_time
outcome_entry_time     = first executable price after decision_time
```

The primary next-24-hour outcome is measured from `outcome_entry_time`, not from a closing price already consumed by the detector.

Default in `configs/baseline.yaml`: close at t is known after the bar ends, so `feature_available_lag_hours: 1`. Cortex uses the same default as a **conservative configured lag** because publication delay is unproven (`docs/CORTEX_TIMING.md`).

## Live-safe state vs one-time event

Stored separately on every curve/hour:

| Column | Meaning |
|---|---|
| `transition_state_raw` | Persistent 0 = upward after last confirmed trough; 1 = downward after last confirmed peak; null until the first confirmation |
| `transition_direction` | +1 / −1 once a turn has been confirmed |
| `transition_event` | `TROUGH_CONFIRMED` / `PEAK_CONFIRMED` on the confirmation row only; `NONE` otherwise |
| `confirmation_time` | Timestamp of the last confirmation (null until the first) |
| `{h}_candidate_extremum_time` | Metadata from confirmation onward (“the detector at 4:00 says the candidate was 2:00”). Null on 2:00 and 3:00. Not a model input |

`transition_event` occurs once. `transition_state_raw` persists until the next confirmed opposite turn. The start of the dataset stays null until the first confirmed turn. No guess, no backward fill.

The timestamp rule (confirm at available time, never rewrite earlier rows) is frozen. The **detector that decides what counts as a turn** is versioned and may change. v1 was delay-only local extrema and fired on small wiggles, especially when T was taken from V. v2 confirms extrema of **D** only, with min-duration and MAD hysteresis.

## Research-only oracle table

File: `oracle_extrema_research_only.csv`. Never merged into the vector or the inference path.

Question: how much of the move was already over by the time the turn became confirmable?

Hindsight extrema may be used only for this waiting-cost study, or later as the *target* of a separate turning-point prediction experiment. They must not be inputs to the 24-hour return model, the production vector, analogue matching, or feature selection.

## Feature allowlist

`market_motion/feature_allowlist.py` is the only numeric-column selector for vectors, analogues, and future models. Oracle columns, waiting-cost fields, candidate timestamps, confirmation events, and outcomes are excluded.

## Same rule everywhere

There is one point-in-time rule. Do not train on hindsight extrema and test on confirmation time.
