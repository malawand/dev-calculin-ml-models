"""Explicit live-feature allowlist. Oracle and outcome columns are never inputs."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .exceptions import VectorSourceError

ORACLE_DEFINITION_VERSION = "oracle_waiting_cost_v1"

ORACLE_EXACT_COLUMNS = frozenset(
    {
        "oracle_extremum_time",
        "oracle_extremum_type",
        "return_from_oracle_extremum",
        "return_from_confirmation",
        "waiting_cost",
        "oracle_definition_version",
        "return_while_waiting",
    }
)

VECTOR_IDENTITY_COLUMNS = frozenset(
    {
        "time_bucket",
        "observation_time_utc",
        "feature_available_time_utc",
        "decision_time_utc",
        "outcome_entry_time_utc",
        "asset",
        "sampling_frequency",
        "vector_source",
        "vector_version",
        "source_query_version",
        "feature_configuration_hash",
        "detector_version",
        "confirmation_delay_hours",
        "min_prominence_mad",
        "min_duration_horizon_fraction",
        "btc_close_at_t",
        "price_source_age_hours",
        "price_feature_valid",
        "price_max_staleness_hours",
        "feature_available_lag_hours",
        "decision_latency_hours",
        "cross_n_horizons_up",
        "cross_n_horizons_valid",
        "row_vector_valid",
        "timing_certainty",
        "source_semantic_certainty",
        "selected_value_rule",
    }
)

# Live-safe per-horizon suffixes. candidate_extremum_time is metadata from
# confirmation onward; it is never a model/analogue input.
PER_HORIZON_LIVE_SUFFIXES = (
    "derivative_raw",
    "position_z",
    "velocity",
    "acceleration",
    "va_stable",
    "ols_n_obs",
    "ols_window_bars",
    "resolution_class",
    "transition_state_raw",
    "transition_raw",
    "transition_direction",
    "transition_event",
    "confirmation_time",
    "hours_since_confirmation",
    "hours_since_turn",
    "candidate_extremum_time",
    "min_duration_hours",
    "velocity_transition_state_raw",
    "velocity_transition_event",
    "velocity_hours_since_confirmation",
    "horizon_hours",
    "promql",
    "source_present",
    "source_age_hours",
    "normalization_ready",
    "feature_valid",
    "invalidation_reason",
    "source_cluster",
    "source_replica",
    "n_candidates",
    "source_conflict",
    "source_value_disagree",
    "source_boundary",
    "source_segment_id",
    "selected_value_rule",
)

MODEL_FEATURE_SUFFIXES = frozenset(
    {
        "derivative_raw",
        "position_z",
        "velocity",
        "acceleration",
        "transition_state_raw",
        "hours_since_confirmation",
    }
)

MODEL_FEATURE_GLOBAL = frozenset(
    {
        "cross_n_horizons_up",
        "cross_n_horizons_valid",
    }
)

METADATA_NOT_MODEL = frozenset(
    {
        "candidate_extremum_time",
        "confirmation_time",
        "transition_event",
        "resolution_class",
        "invalidation_reason",
        "source_cluster",
        "source_replica",
        "n_candidates",
        "source_conflict",
        "source_value_disagree",
        "source_boundary",
        "source_segment_id",
        "selected_value_rule",
        "promql",
    }
)


def _horizon_col(horizon: str, suffix: str) -> str:
    return f"{horizon}_{suffix}"


def live_vector_columns(horizons: Iterable[str]) -> set[str]:
    cols = set(VECTOR_IDENTITY_COLUMNS)
    for h in horizons:
        for suffix in PER_HORIZON_LIVE_SUFFIXES:
            cols.add(_horizon_col(h, suffix))
    return cols


def model_feature_allowlist(horizons: Iterable[str]) -> set[str]:
    cols = set(MODEL_FEATURE_GLOBAL)
    for h in horizons:
        for suffix in MODEL_FEATURE_SUFFIXES:
            cols.add(_horizon_col(h, suffix))
    return cols


def is_oracle_column(name: str) -> bool:
    n = str(name)
    if n in ORACLE_EXACT_COLUMNS:
        return True
    if n.startswith("oracle_"):
        return True
    if n.endswith("_waiting_cost") or n == "waiting_cost":
        return True
    if n.startswith("return_from_oracle") or n.startswith("return_from_confirmation"):
        return True
    return False


def oracle_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if is_oracle_column(c)]


def assert_no_oracle_columns(columns: Iterable[str], *, context: str = "") -> None:
    leaked = oracle_columns(columns)
    if leaked:
        raise VectorSourceError(
            f"{context}oracle/research-only columns are not allowed as live features: {leaked!r}"
        )


def infer_horizons(columns: Iterable[str]) -> list[str]:
    found: list[str] = []
    for c in columns:
        name = str(c)
        if name.endswith("_velocity_transition_state_raw"):
            continue
        if name.endswith("_transition_state_raw"):
            found.append(name[: -len("_transition_state_raw")])
        elif name.endswith("_transition_raw") and not name.endswith("_transition_state_raw"):
            h = name[: -len("_transition_raw")]
            if h not in found:
                found.append(h)
    return found


def select_model_feature_columns(
    df: pd.DataFrame,
    horizons: Iterable[str] | None = None,
) -> list[str]:
    """Numeric live inputs for analogues and future models. Explicit allowlist only."""
    assert_no_oracle_columns(df.columns, context="select_model_feature_columns: ")
    hs = list(horizons) if horizons is not None else infer_horizons(df.columns)
    allowed = model_feature_allowlist(hs)
    selected = [c for c in df.columns if c in allowed]
    for c in selected:
        if "candidate_extremum" in c or c.endswith("_confirmation_time") or c.endswith("_transition_event"):
            raise VectorSourceError(f"metadata column leaked into model allowlist: {c}")
        if c.endswith("_transition_raw") and not c.endswith("_transition_state_raw"):
            raise VectorSourceError(f"alias leaked into model allowlist: {c}")
        if c.endswith("_transition_direction") or c.endswith("_hours_since_turn"):
            raise VectorSourceError(f"duplicate encoding leaked into model allowlist: {c}")
        if "velocity_transition" in c:
            raise VectorSourceError(f"velocity turns are not canonical model features: {c}")
        if any(c.endswith(f"_{s}") for s in METADATA_NOT_MODEL):
            raise VectorSourceError(f"provenance/metadata leaked into model allowlist: {c}")
    return selected


def assert_vector_has_no_oracle(df: pd.DataFrame, *, context: str = "") -> None:
    assert_no_oracle_columns(df.columns, context=context)
    leaked = [c for c in df.columns if "waiting_cost" in str(c) or str(c).startswith("oracle_")]
    if leaked:
        raise VectorSourceError(f"{context}vector artifact contains oracle columns {leaked!r}")
