from typing import Tuple
import numpy as np

def two_sins_series(
    n_points: int = 70,
    x_start: float = 0.0,
    x_end: float = 70,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates the '2 sins' synthetic series and normalizes to [0, 1].

    f(x) = \frac{sin(5.0x)+0.5sin(8.0x)}{4} + 0.5   
    as per the description in the paper:"""
    x = np.linspace(x_start, x_end, n_points, dtype=float)
    y = ((np.sin(5.0 * x) + 0.5 * np.sin(8.0 * x)) / 4.0) + 0.5

    # normalize to [0, 1] (paper says "(0–1)"; min-max gives [0,1])
    y_min, y_max = float(y.min()), float(y.max())
    if np.isclose(y_max, y_min):
        raise ValueError("Degenerate series: cannot normalize (max == min).")
    y_norm = (y - y_min) / (y_max - y_min)

    return x, y_norm