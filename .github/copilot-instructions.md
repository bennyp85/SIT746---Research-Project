# GitHub Copilot Instructions

## Project Overview
This is a quantum machine learning research project using Python and Qiskit. The project explores the intersection of quantum computing and machine learning algorithms.

## Coding Style

### General Principles
- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Keep functions focused and concise (ideally under 50 lines)
- Add docstrings to all functions, classes, and modules
- Use type hints for function signatures

### Python Standards
```python
# Good example
def calculate_quantum_state(circuit: QuantumCircuit, backend: str = 'qasm_simulator') -> np.ndarray:
    """
    Calculate the quantum state vector for a given circuit.
    
    Args:
        circuit: The quantum circuit to execute
        backend: The backend simulator to use
        
    Returns:
        np.ndarray: The state vector representation
    """
    pass
```

### Formatting
- Use Black for code formatting (line length: 100)
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use double quotes for strings

## Naming Conventions

### Files and Directories
- Use snake_case for Python files: `quantum_classifier.py`
- Use lowercase with hyphens for config files: `experiment-config.yml`
- Use descriptive names that indicate purpose

### Variables and Functions
- `snake_case` for variables and functions: `num_qubits`, `run_experiment()`
- `PascalCase` for classes: `QuantumNeuralNetwork`, `DataProcessor`
- `UPPER_CASE` for constants: `MAX_ITERATIONS`, `DEFAULT_BACKEND`

### Quantum-Specific Conventions
- Prefix quantum circuits with `qc_`: `qc_classifier`
- Prefix quantum registers with `qr_`: `qr_data`
- Prefix classical registers with `cr_`: `cr_output`
- Use `n_` prefix for counts: `n_qubits`, `n_layers`

## Project Structure

### Source Code Organization
- `src/quantum_ml/`: Core quantum ML algorithms and circuits
- `src/data/`: Data loading, preprocessing, and feature engineering
- `src/experiments/`: Experiment runners and orchestration
- `tests/`: Unit tests mirroring the src/ structure
- `config/`: Configuration files for experiments
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `docs/`: Documentation, literature reviews, and experiment logs

### Import Order
1. Standard library imports
2. Third-party imports (NumPy, Pandas, etc.)
3. Qiskit imports
4. Local application imports

```python
import os
from typing import List, Dict

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers import Backend

from src.quantum_ml.circuits import create_feature_map
from src.data.preprocessing import normalize_data
```

## Testing Policy

### Test Coverage Requirements
- Minimum 80% code coverage for all modules
- 100% coverage for critical quantum circuit components
- All new features must include corresponding tests

### Test Organization
- Mirror the `src/` directory structure in `tests/`
- Name test files with `test_` prefix: `test_quantum_classifier.py`
- Use descriptive test function names: `test_quantum_circuit_initialization()`

### Testing Best Practices
```python
import pytest
from qiskit import QuantumCircuit

def test_quantum_circuit_creation():
    """Test that quantum circuit is created with correct number of qubits."""
    n_qubits = 4
    qc = QuantumCircuit(n_qubits)
    assert qc.num_qubits == n_qubits
    assert qc.num_clbits == 0

def test_invalid_input_raises_error():
    """Test that invalid inputs raise appropriate errors."""
    with pytest.raises(ValueError):
        create_quantum_classifier(n_qubits=-1)
```

### Test Categories
- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test quantum circuit execution with simulators
- **Benchmark tests**: Measure performance of quantum algorithms
- Use markers: `@pytest.mark.slow` for long-running tests

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_quantum_ml/test_circuits.py

# Run only fast tests
pytest -m "not slow"
```

## Documentation Standards

### Docstring Format (Google Style)
```python
def train_quantum_model(data: np.ndarray, labels: np.ndarray, 
                        n_qubits: int = 4, n_layers: int = 2) -> dict:
    """
    Train a variational quantum classifier on the provided data.
    
    Args:
        data: Input feature vectors of shape (n_samples, n_features)
        labels: Target labels of shape (n_samples,)
        n_qubits: Number of qubits in the quantum circuit
        n_layers: Number of variational layers
        
    Returns:
        dict: Training results containing:
            - 'model': Trained model parameters
            - 'accuracy': Final training accuracy
            - 'loss_history': Loss values per iteration
            
    Raises:
        ValueError: If data dimensions don't match n_qubits
        RuntimeError: If training fails to converge
        
    Example:
        >>> data = np.random.rand(100, 4)
        >>> labels = np.random.randint(0, 2, 100)
        >>> results = train_quantum_model(data, labels, n_qubits=4)
        >>> print(f"Accuracy: {results['accuracy']}")
    """
    pass
```

### Module Documentation
Every module should have a docstring explaining its purpose:
```python
"""
Quantum feature map implementations for data encoding.

This module provides various quantum feature map strategies including:
- Amplitude encoding
- Angle encoding
- QAOA-inspired feature maps
"""
```

## Error Handling

### Exception Handling
- Use specific exception types
- Provide informative error messages
- Log errors appropriately

```python
try:
    result = execute_quantum_circuit(qc, backend)
except QiskitError as e:
    logger.error(f"Qiskit execution failed: {e}")
    raise RuntimeError("Failed to execute quantum circuit") from e
```

## Version Control

### Commit Messages
- Use present tense: "Add feature" not "Added feature"
- First line: concise summary (50 chars max)
- Body: detailed explanation if needed
- Reference issues: "Fixes #123"

### Branch Naming
- `feature/quantum-classifier`: New features
- `bugfix/circuit-initialization`: Bug fixes
- `experiment/vqe-optimization`: Experimental work
- `docs/update-readme`: Documentation updates

## Performance Considerations

### Quantum Circuit Optimization
- Minimize circuit depth where possible
- Use transpiler optimization levels appropriately
- Cache compiled circuits when running multiple times
- Consider noise models for realistic simulations

### Classical Code Optimization
- Use NumPy vectorization over loops
- Profile code to identify bottlenecks
- Use appropriate data structures
- Cache expensive computations

## Security and Privacy

### Sensitive Data
- Never commit API tokens or credentials
- Use environment variables for secrets
- Keep experimental data private if required
- Document data handling requirements

## Continuous Integration

- All tests must pass before merging
- Code must pass linting (flake8, pylint)
- Maintain test coverage standards
- Review CI logs for warnings

## Research-Specific Guidelines

### Reproducibility
- Set random seeds: `np.random.seed(42)`
- Document all hyperparameters in config files
- Save experiment configurations with results
- Version control all code used in papers

### Experiment Tracking
- Log all experiment parameters
- Save results with timestamps
- Keep detailed lab notebooks
- Document failed experiments for learning

### Literature References
- Document papers in `docs/literature/`
- Include implementation references in code
- Cite algorithms in docstrings
