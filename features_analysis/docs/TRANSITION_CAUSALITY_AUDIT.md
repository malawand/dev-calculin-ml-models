# Cortex transition causality audit

**Status: operator closed 2026-08-24** (stored values are not edited after ingest)  
**Date:** 2026-08-23, updated 2026-08-24  
**Scope:** Can Cortex 0/1 flags be used as “known at that hour”?

## Questions

| Question | Finding |
|---|---|
| How are peaks and troughs detected? | **Unknown in this repo.** This checkout only *consumes* Prometheus series `derivative_transitions_2`, `derivative_prime_transitions_2`, and `polynomial_transition`. The detector that writes those series is not here. |
| Does confirmation require bars after the apparent extremum? | **Unknown.** Occupancy scripts (`rank_peak_trough.py`) use a *local* 3-bar sign flip on OLS derivatives. That is **not** the Cortex generator. |
| At what timestamp does a transition first become knowable? | **Unknown for Cortex.** Local OLS canonical T is a confirmed peak/trough of **D** at confirmation time (`causal_d_extrema_*_v2`). Earlier rows are not rewritten. |
| Is the transition backdated to the extremum? | **Unknown for Cortex.** Local OLS `T` is **not** backdated. Candidate extremum time is stored only from the confirmation row onward, as metadata, not as a rewrite of 2:00/3:00 features. |
| Can historical Cortex query results contain retrospectively revised states? | **Operator 2026-08-24: no.** Once ingested, Cortex values are not changed. |

## What this repo does prove

- Downstream Cortex **ingestion** is chronological: hourly UTC grid, no bfill, bounded ffill (`quality.bounded_ffill`).
- Local OLS transitions used in `price_ols_proxy_v1` pass prefix-parity and no-backdating tests (`test_confirmation_timestamps.py`, `test_hourly_grid.py`).
- Digit convention, when bits are present: `transition_state_raw=0` upward after trough; `=1` downward after peak. Signed form: `transition_direction=+1 / -1`. Persistent state is separate from one-row `transition_event`. Initial state is null until the first confirmation.

## What this repo does not prove

Chronological *processing* of a remote series does **not** make the series causal. If Cortex confirms a peak with future bars and then backfills the 0/1 bit onto earlier timestamps, a `query_range` dump would leak.

## Verdict (updated 2026-08-24)

The owner’s rule: **Cortex does not rewrite numbers after they are ingested.** We will treat stored 0/1 flags as immutable.

The remaining (smaller) question is only: when Cortex *first writes* a flag, is the hour on that row the hour it became known, or an earlier peak/trough hour? That is ingest labeling, not later editing. We do not have the generator code in this repo. We will not block the project on it unless analogue results look suspiciously perfect at turning points.

Do not mix Cortex 0/1 bits, Cortex continuous D, or `price_ols_proxy_v1`. They are different measurements.

Continuous Cortex D is now consumed as `cortex_market_motion_v1` with the **local** confirmation-time detector v2. That does not certify the upstream recording-rule timestamp. See `docs/CORTEX_TIMING.md`.
