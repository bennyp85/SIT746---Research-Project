"""
Quantum circuit definitions and utilities.

This module contains functions for creating and manipulating quantum circuits
for machine learning applications.
"""

from qiskit import QuantumCircuit, QuantumRegister


def create_feature_map(
    n_qubits: int, feature_dimension: int, entanglement: str = "full", reps: int = 2
) -> QuantumCircuit:
    """
    Create a quantum feature map for encoding classical data.

    Args:
        n_qubits: Number of qubits in the circuit
        feature_dimension: Dimension of the input feature vector
        entanglement: Type of entanglement ('full', 'linear', 'circular')
        reps: Number of repetitions of the feature map circuit

    Returns:
        QuantumCircuit: The feature map circuit

    Raises:
        ValueError: If n_qubits < 1 or feature_dimension < 1

    Example:
        >>> fm = create_feature_map(n_qubits=4, feature_dimension=4)
        >>> print(f"Circuit has {fm.num_qubits} qubits")
        Circuit has 4 qubits
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be at least 1")

    qr = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(qr, name="FeatureMap")

    # Simple implementation - in practice, use Qiskit's ZZFeatureMap
    # This is a placeholder demonstrating the structure
    # The hardcoded rotation angle (0.1) would be replaced with actual feature values
    for _ in range(reps):
        for i in range(n_qubits):
            qc.h(i)
            qc.rz(0.1, i)  # In real implementation, use input features

    return qc


def create_ansatz(n_qubits: int, n_layers: int = 1) -> QuantumCircuit:
    """
    Create a variational ansatz circuit.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers

    Returns:
        QuantumCircuit: The ansatz circuit
    """
    qr = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(qr, name="Ansatz")

    for layer in range(n_layers):
        # Rotation layer - in practice, these would be trainable parameters
        for i in range(n_qubits):
            qc.ry(0.1, i)  # Placeholder angle; parameters would be optimized

        # Entanglement layer
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc
