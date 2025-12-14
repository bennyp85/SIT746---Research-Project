from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any

try:
    import numpy as np
except ModuleNotFoundError:  # optional dependency for environments without numpy
    np = None

try:
    from scipy.interpolate import interp1d
except ModuleNotFoundError:  # optional dependency for environments without scipy
    interp1d = None
import codecs
from collections import Counter


def tsf_series_lengths(path):
    lengths = []
    names = []
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
            # Format: series_name : start_timestamp : v1,v2,...,vN
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            name, start_ts, values_str = parts
            values = [v for v in values_str.split(",") if v != ""]
            names.append(name)
            lengths.append(len(values))
    return names, lengths

def tsf_series_load(path, names=None):
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
            name, start_ts, values_str = parts
            if names is not None and name not in names:
                continue
            values = [float(v) for v in values_str.split(",") if v != ""]
            if np is None:
                series_list.append(values)
            else:
                series_list.append(np.array(values))
    return series_list

def uniform_resample(series, target_len):
    if np is None:
        raise ModuleNotFoundError("numpy is required for uniform_resample()")
    if interp1d is None:
        raise ModuleNotFoundError("scipy is required for uniform_resample()")

    x = np.linspace(0, 1, len(series))
    f = interp1d(x, series, kind="linear")
    x_new = np.linspace(0, 1, target_len)
    return f(x_new)



def compute_paa_series(series, n_segments: int = 20):
    if series is None:
        raise ValueError("series must not be None")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")

    if np is None:
        values = [float(x) for x in series]
        n = len(values)
        if n == 0:
            raise ValueError("series must not be empty")
        if n_segments > n:
            raise ValueError("n_segments cannot exceed the series length")

        out = []
        for i in range(n_segments):
            start = (i * n) // n_segments
            end = ((i + 1) * n) // n_segments
            if end <= start:
                end = min(start + 1, n)
            segment = values[start:end]
            out.append(sum(segment) / len(segment))
        return out

    series_arr = np.asarray(series, dtype=float).reshape(-1)
    n = int(series_arr.size)
    if n == 0:
        raise ValueError("series must not be empty")
    if n_segments > n:
        raise ValueError("n_segments cannot exceed the series length")

    boundaries = np.linspace(0, n, num=n_segments + 1)
    starts = np.floor(boundaries[:-1]).astype(int)
    ends = np.floor(boundaries[1:]).astype(int)
    ends = np.maximum(ends, starts + 1)
    ends = np.minimum(ends, n)
    starts = np.minimum(starts, n - 1)

    out = np.empty(n_segments, dtype=float)
    for i, (start, end) in enumerate(zip(starts, ends)):
        out[i] = float(series_arr[start:end].mean())
    return out


def _paa_segment_bounds(n: int, n_segments: int) -> List[Tuple[int, int]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    if n_segments > n:
        raise ValueError("n_segments cannot exceed the series length")

    if np is None:
        bounds: List[Tuple[int, int]] = []
        for i in range(n_segments):
            start = (i * n) // n_segments
            end = ((i + 1) * n) // n_segments
            if end <= start:
                end = min(start + 1, n)
            bounds.append((start, end))
        return bounds

    boundaries = np.linspace(0, n, num=n_segments + 1)
    starts = np.floor(boundaries[:-1]).astype(int)
    ends = np.floor(boundaries[1:]).astype(int)
    ends = np.maximum(ends, starts + 1)
    ends = np.minimum(ends, n)
    starts = np.minimum(starts, n - 1)
    return list(zip(starts.tolist(), ends.tolist()))


def compute_paa_trend_segments(series, n_segments: int = 20):
    """Compute per-PAA-segment trend deltas using each segment's first and last values.

    Returns:
      - trend_deltas: shape (n_segments,) array (or list if numpy missing)
      - segments: list of dicts with start/end/first/last/delta per segment
    """
    if series is None:
        raise ValueError("series must not be None")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")

    if np is None:
        values = [float(x) for x in series]
        n = len(values)
        if n == 0:
            raise ValueError("series must not be empty")
        bounds = _paa_segment_bounds(n, n_segments)

        deltas: List[float] = []
        segments: List[Dict[str, Any]] = []
        for i, (start, end) in enumerate(bounds):
            first = float(values[start])
            last = float(values[end - 1])
            delta = last - first
            deltas.append(delta)
            segments.append(
                {
                    "segment": int(i),
                    "start": int(start),
                    "end": int(end),
                    "first": first,
                    "last": last,
                    "delta": delta,
                }
            )
        return deltas, segments

    series_arr = np.asarray(series, dtype=float).reshape(-1)
    series_arr = series_arr[np.isfinite(series_arr)]
    n = int(series_arr.size)
    if n == 0:
        raise ValueError("series must not be empty")
    bounds = _paa_segment_bounds(n, n_segments)

    deltas = np.empty(n_segments, dtype=float)
    segments: List[Dict[str, Any]] = []
    for i, (start, end) in enumerate(bounds):
        first = float(series_arr[start])
        last = float(series_arr[end - 1])
        delta = last - first
        deltas[i] = delta
        segments.append(
            {
                "segment": int(i),
                "start": int(start),
                "end": int(end),
                "first": first,
                "last": last,
                "delta": float(delta),
            }
        )
    return deltas, segments


def paa_trend_directions(
    series,
    window_size: int,
    threshold: float = 0.1,
    num_segments: Optional[int] = None,
) -> List[int]:
    if series is None:
        raise ValueError("series must not be None")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    n = len(series)
    if n == 0:
        raise ValueError("series must not be empty")

    if num_segments is None:
        num_segments = (n + window_size - 1) // window_size
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")

    directions: List[int] = []
    for i in range(num_segments):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, n)
        if start_idx >= n:
            break

        segment = series[start_idx:end_idx]
        if len(segment) < 2:
            trend = 0.0
        else:
            trend = float(segment[-1]) - float(segment[0])

        if trend > threshold:
            directions.append(1)
        elif trend < -threshold:
            directions.append(-1)
        else:
            directions.append(0)

    return directions
