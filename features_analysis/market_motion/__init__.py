"""Market-motion vector research package (see PROJECT_PLAN.md).

Three labeled artifacts, never mixed:

* ``cortex_transition_baseline_v1`` — binary Cortex 0/1 transitions (Phase 1)
* ``price_ols_proxy_v1`` — causal OLS on log close (Vector v1 proxy)
* ``cortex_market_motion_v1`` — continuous Cortex D (8h, not 7h; no OLS fallback)

OLS is price trend/momentum. It is not a fallback for missing Cortex derivatives.
"""

__version__ = "0.3.0"

BASELINE_VERSION = "cortex_transition_baseline_v1"

VECTOR_SOURCE_OLS = "price_ols_proxy"
VECTOR_VERSION_OLS = "price_ols_proxy_v1"

VECTOR_SOURCE_CORTEX_MOTION = "cortex_market_motion"
VECTOR_VERSION_CORTEX_MOTION = "cortex_market_motion_v1"

# Deprecated names from the first draft of this branch. Using them is an error.
DEPRECATED_VECTOR_SOURCES = {
    "cortex_derivative": VECTOR_SOURCE_CORTEX_MOTION,
    "market_motion_v1_price_ols_proxy": VECTOR_VERSION_OLS,
    "market_motion_v1_cortex_derivative": VECTOR_VERSION_CORTEX_MOTION,
}

OUTCOME_VERSION = "btc_outcomes_v1"

# Cortex / local OLS transition convention (not sign(D)).
TRANSITION_RAW_UP_AFTER_TROUGH = 0.0
TRANSITION_RAW_DOWN_AFTER_PEAK = 1.0
TRANSITION_DIRECTION_UP = 1.0
TRANSITION_DIRECTION_DOWN = -1.0

TIMESTAMP_COLS = (
    "observation_time_utc",
    "feature_available_time_utc",
    "decision_time_utc",
    "outcome_entry_time_utc",
)
TIMESTAMP_COLS = TIMESTAMP_COLS
