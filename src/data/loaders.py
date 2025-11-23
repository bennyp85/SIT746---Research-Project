"""
Data loading utilities for quantum ML experiments.

This module provides functions to load and prepare datasets for quantum
machine learning experiments.
"""

from typing import Tuple
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_sample_data(
    dataset: str = "iris", test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a sample dataset for testing quantum ML algorithms.

    Args:
        dataset: Name of the dataset ('iris', 'synthetic')
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Example:
        >>> X_train, X_test, y_train, y_test = load_sample_data('iris')
        >>> print(f"Training samples: {len(X_train)}")
        Training samples: 120
    """
    if dataset == "iris":
        data = load_iris()
        X, y = data.data, data.target
        # Binary classification: class 0 vs rest
        y = (y == 0).astype(int)
    elif dataset == "synthetic":
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Split and normalize
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
