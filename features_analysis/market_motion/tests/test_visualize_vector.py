"""Visualization helpers: confirmation-time markers, no outcomes, T greying."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_motion.transition import EVENT_NONE, EVENT_PEAK, EVENT_TROUGH
from market_motion.visualize_vector import (
    NEEDED,
    describe_row,
    downsample_for_html,
    event_masks,
    heatmap_matrix,
    masked_state,
)


def _row(**overrides):
    rec = {
        "time_bucket": pd.Timestamp("2024-01-10 12:00", tz="UTC"),
        "btc_close_at_t": 42000.0,
        "cross_n_horizons_up": 6,
        "cross_n_horizons_valid": 7,
    }
    for h in ("2h", "4h", "8h", "12h", "24h", "48h", "7d"):
        rec[f"{h}_feature_valid"] = True
        rec[f"{h}_transition_state_raw"] = 0.0
        rec[f"{h}_position_z"] = 0.4
        rec[f"{h}_velocity"] = 0.1
        rec[f"{h}_derivative_raw"] = 1.0
        rec[f"{h}_acceleration"] = 0.0
        rec[f"{h}_hours_since_confirmation"] = 3.0
        rec[f"{h}_transition_event"] = EVENT_NONE
    rec.update(overrides)
    return rec


def test_needed_columns_exclude_oracle_and_candidates():
    joined = " ".join(NEEDED)
    assert "candidate_extremum" not in joined
    assert "oracle" not in joined
    assert "return" not in joined
    assert "crps" not in joined


def test_describe_row_is_not_a_forecast():
    row = pd.Series(_row(**{"2h_position_z": 3.2, "2h_velocity": 0.8, "8h_transition_state_raw": 1.0, "8h_velocity": -0.4}))
    text = describe_row(row)
    assert "6 of 7 valid horizons are upward after a trough" in text
    assert "The 2h curve is upward after a trough unusually elevated and rising" in text
    assert "The 8h curve is downward after a peak" in text
    assert "not a forecast" in text.lower()
    assert "buy" not in text.lower()
    assert "sell" not in text.lower()


def test_invalid_horizon_is_grey_in_t_heatmap():
    df = pd.DataFrame(
        [
            _row(**{"2h_feature_valid": False, "2h_transition_state_raw": 0.0}),
            _row(time_bucket=pd.Timestamp("2024-01-10 13:00", tz="UTC")),
        ]
    )
    color, raw = heatmap_matrix(df, "T")
    assert np.isnan(color[0, 0])
    assert np.isnan(raw[0, 0])
    assert color[0, 1] == 0.0
    assert np.isnan(masked_state(df, "2h")[0])


def test_event_masks_use_confirmation_not_row_order():
    df = pd.DataFrame(
        [
            _row(
                time_bucket=pd.Timestamp("2024-01-10 03:00", tz="UTC"),
                **{"2h_transition_event": EVENT_NONE},
            ),
            _row(
                time_bucket=pd.Timestamp("2024-01-10 05:00", tz="UTC"),
                **{"2h_transition_event": EVENT_PEAK},
            ),
            _row(
                time_bucket=pd.Timestamp("2024-01-10 08:00", tz="UTC"),
                **{"2h_transition_event": EVENT_TROUGH},
            ),
        ]
    )
    peaks, troughs = event_masks(df, "2h")
    assert list(peaks) == [False, True, False]
    assert list(troughs) == [False, False, True]


def test_html_downsample_keeps_later_confirmed_state():
    rows = [_row(time_bucket=pd.Timestamp("2024-01-10 03:00", tz="UTC") + pd.Timedelta(hours=i)) for i in range(4)]
    rows[1]["2h_transition_event"] = EVENT_PEAK
    rows[1]["2h_transition_state_raw"] = 1.0
    rows[2]["2h_transition_state_raw"] = 1.0
    rows[3]["2h_transition_state_raw"] = 1.0
    df = pd.DataFrame(rows)
    slim = downsample_for_html(df, hours=2)
    assert len(slim) == 2
    assert slim["2h_transition_state_raw"].iloc[1] == 1.0
