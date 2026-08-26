"""Refuse mixed vector sources and outcome leakage into inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import TIMESTAMP_COLS, VECTOR_SOURCE_CORTEX_MOTION, VECTOR_SOURCE_OLS
from .exceptions import VectorSourceError
from .feature_allowlist import assert_no_oracle_columns, select_model_feature_columns
from .oracle import ORACLE_FILE_NAME

OUTCOME_PREFIXES = (
    "return_",
    "log_return_",
    "label_up_down_flat_",
    "close_path_mfe_",
    "close_path_mae_",
    "time_to_close_path_",
)

RESEARCH_VIEW_NAMES = frozenset(
    {"research_view.csv", "research_view", "joined_research.csv"}
)


def assert_single_vector_identity(df: pd.DataFrame, *, context: str = "") -> tuple[str, str]:
    if "vector_source" not in df.columns or "vector_version" not in df.columns:
        raise VectorSourceError(
            f"{context}vector frame must carry vector_source and vector_version"
        )
    sources = [s for s in df["vector_source"].dropna().unique().tolist()]
    versions = [s for s in df["vector_version"].dropna().unique().tolist()]
    if len(sources) != 1 or len(versions) != 1:
        raise VectorSourceError(
            f"{context}mixed vector identity: sources={sources!r} versions={versions!r}. "
            "price_ols_proxy_v1 and cortex_market_motion_v1 are different hypotheses."
        )
    for col in ("source_query_version", "detector_version", "feature_configuration_hash"):
        if col not in df.columns:
            continue
        vals = [s for s in df[col].dropna().unique().tolist()]
        if len(vals) > 1:
            raise VectorSourceError(
                f"{context}mixed {col}: {vals!r}. Inference must not mix vector configurations."
            )
    return str(sources[0]), str(versions[0])


def concat_vector_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise VectorSourceError("no vector frames to concatenate")
    identities = [assert_single_vector_identity(f) for f in frames]
    if len(set(identities)) != 1:
        raise VectorSourceError(
            f"refusing to concatenate incompatible vector artifacts: {identities}"
        )
    return pd.concat(frames, ignore_index=True)


def outcome_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith(OUTCOME_PREFIXES)]


def load_inference_vector(path: str | Path) -> pd.DataFrame:
    """Load a vector artifact for inference. Never load the joined research view."""
    p = Path(path)
    if p.name in RESEARCH_VIEW_NAMES or "research_view" in p.name:
        raise VectorSourceError(
            f"inference must not load {p.name}; pass a vector CSV such as "
            "vector_price_ols_proxy_v1.csv or vector_cortex_market_motion_v1.csv "
            "(features only, no outcomes)"
        )
    if p.name == ORACLE_FILE_NAME or "oracle" in p.name:
        raise VectorSourceError(
            f"inference must not load {p.name}; oracle extrema are research-only"
        )
    df = pd.read_csv(p)
    leaked = outcome_columns(df)
    if leaked:
        raise VectorSourceError(
            f"inference vector {p} contains outcome columns {leaked[:6]!r}; "
            "load the vector file, not the research view"
        )
    assert_no_oracle_columns(df.columns, context=f"{p}: ")
    source, version = assert_single_vector_identity(df, context=f"{p}: ")
    if source == VECTOR_SOURCE_CORTEX_MOTION:
        if version != "cortex_market_motion_v1":
            raise VectorSourceError(f"unsupported cortex vector_version={version!r}")
    elif source != VECTOR_SOURCE_OLS:
        raise VectorSourceError(f"unsupported inference vector_source={source!r}")
    # Touch the allowlist so automatic numeric selection cannot silently
    # include metadata. Result is unused here; analogue/model code must call it.
    select_model_feature_columns(df)
    return df


def inference_row_is_actionable(row: pd.Series) -> bool:
    valid = row.get("row_vector_valid")
    if valid is None or (isinstance(valid, float) and pd.isna(valid)):
        return False
    return bool(valid)


def timing_skip_columns() -> set[str]:
    return {"time_bucket", *TIMESTAMP_COLS}
