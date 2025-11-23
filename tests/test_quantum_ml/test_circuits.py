"""
Test suite for quantum circuit implementations.
"""

import pytest
from qiskit import QuantumCircuit
from src.quantum_ml.circuits import create_feature_map, create_ansatz


def test_create_feature_map_basic():
    """Test basic feature map creation."""
    n_qubits = 4
    feature_dim = 4
    fm = create_feature_map(n_qubits, feature_dim)
    
    assert isinstance(fm, QuantumCircuit)
    assert fm.num_qubits == n_qubits
    assert fm.name == 'FeatureMap'


def test_create_feature_map_invalid_qubits():
    """Test that invalid qubit count raises ValueError."""
    with pytest.raises(ValueError, match="n_qubits must be at least 1"):
        create_feature_map(n_qubits=0, feature_dimension=4)


def test_create_feature_map_invalid_dimension():
    """Test that invalid feature dimension raises ValueError."""
    with pytest.raises(ValueError, match="feature_dimension must be at least 1"):
        create_feature_map(n_qubits=4, feature_dimension=0)


def test_create_ansatz_basic():
    """Test basic ansatz creation."""
    n_qubits = 4
    n_layers = 2
    ansatz = create_ansatz(n_qubits, n_layers)
    
    assert isinstance(ansatz, QuantumCircuit)
    assert ansatz.num_qubits == n_qubits
    assert ansatz.name == 'Ansatz'


def test_create_ansatz_single_layer():
    """Test ansatz with single layer."""
    ansatz = create_ansatz(n_qubits=2, n_layers=1)
    assert ansatz.num_qubits == 2


@pytest.mark.parametrize("n_qubits,n_layers", [
    (2, 1),
    (4, 2),
    (8, 3),
])
def test_create_ansatz_various_sizes(n_qubits, n_layers):
    """Test ansatz creation with various sizes."""
    ansatz = create_ansatz(n_qubits, n_layers)
    assert ansatz.num_qubits == n_qubits
