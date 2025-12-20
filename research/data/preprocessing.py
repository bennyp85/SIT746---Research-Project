import numpy as np

def paa(y: np.ndarray, w: int) -> np.ndarray:
    """
    Piecewise Aggregate Approximation (PAA)

    Parameters
    ----------
    y : np.ndarray
        1D time series of length n
    w : int
        Number of PAA segments

    Returns
    -------
    np.ndarray
        PAA representation of length w
    """
    n = len(y)
    return np.array([
        y[int(i*n/w):int((i+1)*n/w)].mean()
        for i in range(w)
    ])
