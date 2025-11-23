"""
Test suite for data loading utilities.
"""

import pytest
import numpy as np
from src.data.loaders import load_sample_data


def test_load_iris_data():
    """Test loading iris dataset."""
    X_train, X_test, y_train, y_test = load_sample_data('iris')
    
    # Check shapes
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == 4  # Iris has 4 features
    
    # Check labels are binary
    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})
    
    # Check data is normalized (approximately)
    assert np.abs(X_train.mean()) < 0.5
    assert np.abs(X_train.std() - 1.0) < 0.5


def test_load_synthetic_data():
    """Test loading synthetic dataset."""
    X_train, X_test, y_train, y_test = load_sample_data('synthetic')
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == 4


def test_load_unknown_dataset():
    """Test that unknown dataset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_sample_data('unknown')


def test_data_split_ratio():
    """Test that data split respects test_size parameter."""
    test_size = 0.3
    X_train, X_test, y_train, y_test = load_sample_data(
        'iris', test_size=test_size
    )
    
    total_samples = len(X_train) + len(X_test)
    actual_test_ratio = len(X_test) / total_samples
    
    # Allow some tolerance due to rounding
    assert abs(actual_test_ratio - test_size) < 0.05
