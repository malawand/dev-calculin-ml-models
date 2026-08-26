"""Explicit failures — never silent OLS fallback for Cortex motion."""


class CortexMotionNotImplementedError(NotImplementedError):
    """Raised when the OLS engine is asked to build cortex_market_motion_v1.

    OLS must refuse rather than substitute price slopes for missing Cortex D.
    """


class CortexSourceError(ValueError):
    """Missing, empty, or failed Cortex continuous-derivative source."""


class VectorSourceError(ValueError):
    """Unknown, deprecated, or mixed vector source/version."""
