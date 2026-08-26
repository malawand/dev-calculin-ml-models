# Cortex continuous-D timing audit

**Vector:** `cortex_market_motion_v1`  
**Date:** 2026-08-25  
**Certainty:** `uncertain_recording_rule` (not proven)

This audit is about **when a Cortex sample may be used**, not about whether stored values later change. Operator 2026-08-24: values are not edited after ingest. Immutability after ingest does **not** prove that the first timestamp was the time the feature became knowable.

## Clocks (every continuous series)

The live vector must obey `state_effective_time = feature_available_time`.

| Clock | What we store | What we know |
|---|---|---|
| Source sample timestamp | Prometheus `query_range` sample time (`timestamp` → `time_bucket` / `observation_time_utc`) | Present on the raw tape. Hourly unix times in this dump. |
| Prometheus / Cortex scrape or evaluation time | Same timestamp in this dump, as far as we can see | We do **not** have scrape-config or recording-rule eval time separate from the sample. |
| `feature_available_time` | `observation_time_utc + feature_available_lag_hours` (configured **1h**) | Conservative lag because publication delay is unproven. |
| `decision_time` | `feature_available_time + decision_latency_hours` (0) | Same as available time in this version. |
| `outcome_entry_time` | Same as `decision_time` | First executable close **after** the observation close. No same-close execution. |

A simulated outcome cannot begin at a price already used to calculate or retrieve the feature.

## What the source timestamp likely is

Best **unproven** interpretation: the sample time is a **recording-rule evaluation time**, which in Prometheus is usually aligned with the **end** of the lookback the rule used, not the beginning of the window.

It is **not proven** to be:

- the beginning of the measurement window;
- the end of the measurement window; or
- wall-clock publication time after the rule ran.

The recording-rule text is not in this repository. Instant `query` only proves a series exists now. This pass used `query_range` for history.

## Conservative lag

`cortex_motion.feature_available_lag_hours: 1` in `configs/baseline.yaml`.

If a sample is labeled 4:00, the vector treats it as available at 5:00. If the rule actually evaluates at 4:00 using a window ending at 4:00, this lag is conservative. If Cortex publishes later than one hour, this lag is **insufficient** — that would be a remaining blocker if analogue results look too well aligned with turning points.

## Raw vs filled

1. Raw `query_range` rows are saved **before** any fill (`cortex_continuous_derivatives_raw.csv`). Historical reconstruction uses that unfiltered dump; live/future fetch injects `cluster="PROD-A-crypto-metrics"`.
2. Same-timestamp signed values are selected with `prod_a_preferred_v1` after that save. `max()` is not used for signed Cortex D. The highest-priority **approved** source wins. Competing values and disagreement are recorded in diagnostics. The hour is not marked missing when an approved source is present.
3. True `source_conflict` only when no approved source can be selected. Those hours stay missing.
4. Gaps are **not** backward-filled. A real time gap remains a gap. A routing-label change is not a gap.
5. Bounded forward-fill (2 hours) is applied only when building the vector, cannot cross a true conflict hour, and does not restart because the cluster/replica label changed.
6. Z/V/A and the detector run on the continuous selected series. `source_boundary` / `source_segment_id` are diagnostic provenance only.
7. Cluster/replica are owner-confirmed routing metadata for the same feature family. The upstream derivative formula text remains unavailable in this repository.

## Series covered

`deriv2h`, `deriv4h`, `deriv8h`, `deriv12h`, `deriv24h`, `deriv48h`, `deriv7d`.  
Primes and weighted derivatives are on the tape only. They do not enter Vector v1.
