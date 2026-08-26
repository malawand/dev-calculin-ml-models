"""Refuse mixed vector sources and outcome leakage into inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_motion.exceptions import VectorSourceError
from market_motion.source_guard import (
    concat_vector_frames,
    inference_row_is_actionable,
    load_inference_vector,
)


def test_concat_refuses_mixed_sources():
    a = pd.DataFrame(
        {
            "vector_source": ["price_ols_proxy"],
            "vector_version": ["price_ols_proxy_v1"],
            "x": [1.0],
        }
    )
    b = pd.DataFrame(
        {
            "vector_source": ["cortex_market_motion"],
            "vector_version": ["cortex_market_motion_v1"],
            "x": [2.0],
        }
    )
    with pytest.raises(VectorSourceError, match="incompatible"):
        concat_vector_frames([a, b])


def test_concat_same_source_ok():
    a = pd.DataFrame(
        {
            "vector_source": ["price_ols_proxy"],
            "vector_version": ["price_ols_proxy_v1"],
            "x": [1.0],
        }
    )
    b = a.copy()
    b["x"] = 2.0
    out = concat_vector_frames([a, b])
    assert len(out) == 2


def test_infer_refuses_research_view(tmp_path: Path):
    p = tmp_path / "research_view.csv"
    pd.DataFrame({"vector_source": ["price_ols_proxy"], "return_1h": [0.1]}).to_csv(p, index=False)
    with pytest.raises(VectorSourceError, match="research_view"):
        load_inference_vector(p)


def test_infer_refuses_outcome_columns(tmp_path: Path):
    p = tmp_path / "vector_price_ols_proxy_v1.csv"
    pd.DataFrame(
        {
            "vector_source": ["price_ols_proxy"],
            "vector_version": ["price_ols_proxy_v1"],
            "return_24h": [0.2],
            "row_vector_valid": [True],
        }
    ).to_csv(p, index=False)
    with pytest.raises(VectorSourceError, match="outcome"):
        load_inference_vector(p)


def test_infer_no_prediction_when_invalid(tmp_path: Path):
    p = tmp_path / "vector_price_ols_proxy_v1.csv"
    pd.DataFrame(
        {
            "vector_source": ["price_ols_proxy"],
            "vector_version": ["price_ols_proxy_v1"],
            "row_vector_valid": [False],
            "2h_position_z": [0.1],
        }
    ).to_csv(p, index=False)
    df = load_inference_vector(p)
    assert inference_row_is_actionable(df.iloc[-1]) is False
