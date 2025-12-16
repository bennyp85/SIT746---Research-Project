from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
from collections import Counter
import codecs

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from scipy.interpolate import interp1d
except ModuleNotFoundError:
    interp1d = None


# ============================================================
# TSF LOADING UTILITIES
# ============================================================

def tsf_series_lengths(path: str) -> Tuple[List[str], List[int]]:
    names, lengths = [], []

    with codecs.open(path, encoding="latin-1") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue

            parts = line.split(":", 2)
            if len(parts) != 3:
                continue

            name, _, values_str = parts
            values = [v for v in values_str.split(",") if v != ""]
            names.append(name)
            lengths.append(len(values))

    return names, lengths


def tsf_series_load(path: str, names: Optional[List[str]] = None):
    series_list = []

    with codecs.open(path, encoding="latin-1") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue

            parts = line.split(":", 2)
            if len(parts) != 3:
                continue

            name, _, values_str = parts
            if names is not None and name not in names:
                continue

            values = [float(v) for v in values_str.split(",") if v != ""]
            series_list.append(np.asarray(values) if np is not None else values)

    return series_list


def most_common_length(series_list: List[List[float]]) -> int:
    return Counter(len(s) for s in series_list).most_common(1)[0][0]


# ============================================================
# RESAMPLING
# ============================================================

def uniform_resample(series, target_len: int):
    if np is None or interp1d is None:
        raise ModuleNotFoundError("numpy and scipy required")

    x = np.linspace(0.0, 1.0, len(series))
    f = interp1d(x, series, kind="linear")
    return f(np.linspace(0.0, 1.0, target_len))


# ============================================================
# PAA
# ============================================================

def paa_segment_bounds(n: int, n_segments: int) -> List[Tuple[int, int]]:
    if n_segments <= 0 or n_segments > n:
        raise ValueError("invalid n_segments")

    boundaries = np.linspace(0, n, n_segments + 1)
    starts = np.floor(boundaries[:-1]).astype(int)
    ends = np.floor(boundaries[1:]).astype(int)

    ends = np.maximum(ends, starts + 1)
    ends = np.minimum(ends, n)
    starts = np.minimum(starts, n - 1)

    return list(zip(starts.tolist(), ends.tolist()))


def compute_paa_series(series, n_segments: int) -> np.ndarray:
    series = np.asarray(series, dtype=float).reshape(-1)
    if series.size == 0:
        raise ValueError("empty series")

    bounds = paa_segment_bounds(len(series), n_segments)
    return np.array([series[start:end].mean() for start, end in bounds])


# ============================================================
# TREND COMPUTATION (RAW SPACE)
# ============================================================

def compute_raw_trend_deltas(
    series,
    n_segments: int
) -> np.ndarray:
    """
    Compute raw intra-segment deltas:
        delta_i = last_value - first_value
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    if series.size == 0:
        raise ValueError("empty series")

    bounds = paa_segment_bounds(len(series), n_segments)
    deltas = np.empty(len(bounds), dtype=float)

    for i, (start, end) in enumerate(bounds):
        deltas[i] = series[end - 1] - series[start]

    return deltas


def paa_trend_directions(
    series,
    n_segments: int,
    threshold: float,
) -> List[int]:
    """
    Map raw intra-segment deltas to {-1, 0, +1}
    """
    deltas = compute_raw_trend_deltas(series, n_segments)

    directions: List[int] = []
    for d in deltas:
        if d > threshold:
            directions.append(1)
        elif d < -threshold:
            directions.append(-1)
        else:
            directions.append(0)

    return directions


# ============================================================
# PER-SERIES ADAPTIVE THRESHOLD
# ============================================================

def adaptive_trend_threshold(
    series,
    n_segments: int,
    k: float = 1.0,
) -> float:
    """
    threshold = k * std(raw intra-segment deltas)
    """
    deltas = compute_raw_trend_deltas(series, n_segments)
    return k * float(np.std(deltas))