"""
Data processing and loading utilities for quantum ML experiments.
"""

from .preprocessing import (
    # TSF loading
    tsf_series_lengths,
    tsf_series_load,
    most_common_length,

    # Resampling
    uniform_resample,

    # PAA
    compute_paa_series,
    paa_segment_bounds,

    # Trend logic (raw-space, scale-consistent)
    compute_raw_trend_deltas,
    adaptive_trend_threshold,
    paa_trend_directions,
)

__all__ = [
    # TSF loading
    "tsf_series_lengths",
    "tsf_series_load",
    "most_common_length",

    # Resampling
    "uniform_resample",

    # PAA
    "compute_paa_series",
    "paa_segment_bounds",

    # Trend
    "compute_raw_trend_deltas",
    "adaptive_trend_threshold",
    "paa_trend_directions",
]
