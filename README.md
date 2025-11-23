# Quantum Machine Learning Research Project

[![CI](https://github.com/bennyp85/SIT746---Research-Project/workflows/CI/badge.svg)](https://github.com/bennyp85/SIT746---Research-Project/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A50.45-6133BD)](https://qiskit.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research project exploring the intersection of quantum computing and machine learning, developed as part of the SIT746 Honours Research Project at Deakin University.

## 🎯 Project Overview

This project investigates the application of quantum computing principles to machine learning algorithms, leveraging IBM's Qiskit framework. The research focuses on:

- **Variational Quantum Algorithms (VQA)**: Implementing hybrid quantum-classical algorithms for optimization and machine learning tasks
- **Quantum Neural Networks (QNN)**: Developing and benchmarking quantum circuit-based neural network architectures
- **Quantum Feature Maps**: Exploring different encoding strategies for classical data in quantum states
- **Quantum Kernels**: Investigating quantum kernel methods for classification and regression problems
- **Performance Analysis**: Comparing quantum algorithms with classical baselines on various datasets

## 🛠 Technology Stack

### Quantum Computing
- **Qiskit** (≥0.45.0): IBM's quantum computing framework
- **Qiskit Aer**: High-performance quantum circuit simulators
- **Qiskit Machine Learning**: Quantum ML algorithms and neural networks
- **Qiskit Optimization**: Quantum optimization algorithms

### Machine Learning
- **PyTorch** (≥2.0.0): Deep learning framework for hybrid models
- **TensorFlow** (≥2.13.0): Alternative ML framework
- **scikit-learn** (≥1.3.0): Classical ML algorithms and utilities
- **NumPy** & **SciPy**: Numerical computing libraries

### Data Science & Visualization
- **Pandas**: Data manipulation and analysis
- **Matplotlib** & **Seaborn**: Visualization libraries
- **Jupyter Lab**: Interactive development environment

### Development Tools
- **pytest**: Testing framework with coverage reporting
- **Black**: Code formatter (line length: 100)
- **flake8** & **pylint**: Linting tools
- **mypy**: Static type checker

## 📁 Project Structure

```
SIT746---Research-Project/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── .github/                           # GitHub configuration
│   ├── copilot-instructions.md       # Coding standards and guidelines
│   └── workflows/
│       └── ci.yml                    # Continuous integration pipeline
│
├── docs/                             # Documentation
│   ├── literature/                   # Literature reviews and paper summaries
│   └── experiments/                  # Experiment logs and results analysis
│
├── src/                              # Source code
│   ├── quantum_ml/                   # Core quantum ML implementations
│   │   ├── __init__.py
│   │   ├── circuits.py              # Quantum circuit definitions
│   │   ├── feature_maps.py          # Data encoding strategies
│   │   ├── kernels.py               # Quantum kernel methods
│   │   └── models.py                # QNN and VQA models
│   │
│   ├── data/                         # Data processing
│   │   ├── __init__.py
│   │   ├── loaders.py               # Dataset loading utilities
│   │   └── preprocessing.py         # Data preprocessing and feature engineering
│   │
│   └── experiments/                  # Experiment orchestration
│       ├── __init__.py
│       ├── runner.py                # Experiment execution framework
│       └── utils.py                 # Experiment utilities
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── test_quantum_ml/             # Tests for quantum ML modules
│   ├── test_data/                   # Tests for data processing
│   └── test_experiments/            # Tests for experiment runners
│
├── config/                           # Configuration files
│   ├── default.yml                  # Default configuration
│   └── experiments/                 # Experiment-specific configs
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_exploration.ipynb         # Initial data exploration
│   ├── 02_quantum_circuits.ipynb    # Quantum circuit examples
│   └── 03_experiments.ipynb         # Experiment notebooks
│
└── results/                          # Experiment results
    ├── figures/                      # Generated plots and visualizations
    ├── models/                       # Saved model checkpoints
    └── logs/                         # Experiment logs
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Jupyter Lab for notebooks
- (Optional) IBM Quantum account for real quantum hardware access

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bennyp85/SIT746---Research-Project.git
   cd SIT746---Research-Project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import qiskit; print(qiskit.__version__)"
   pytest tests/ --collect-only  # See available tests
   ```

### Configuration

1. **Set up IBM Quantum account** (optional, for real hardware)
   ```python
   from qiskit import IBMQ
   IBMQ.save_account('YOUR_API_TOKEN')
   ```

2. **Configure experiments**
   - Copy `config/default.yml` to `config/local.yml`
   - Modify parameters as needed
   - Local config is git-ignored for your personal settings

### Quick Start Example

```python
from qiskit import QuantumCircuit
from src.quantum_ml.circuits import create_feature_map
from src.data.loaders import load_sample_data

# Load sample data
X_train, y_train = load_sample_data()

# Create a quantum feature map
n_qubits = 4
feature_map = create_feature_map(n_qubits, feature_dimension=X_train.shape[1])

# Build and visualize the circuit
qc = QuantumCircuit(n_qubits)
qc.compose(feature_map, inplace=True)
print(qc.draw())
```

## 🧪 Running Experiments

### Basic Workflow

1. **Define an experiment configuration** in `config/experiments/`
   ```yaml
   experiment:
     name: "quantum_classifier_benchmark"
     dataset: "iris"
     n_qubits: 4
     n_layers: 2
     backend: "qasm_simulator"
     shots: 1024
   ```

2. **Run the experiment**
   ```bash
   python -m src.experiments.runner --config config/experiments/your_config.yml
   ```

3. **Analyze results** in `notebooks/` or view outputs in `results/`

### Example Experiments

#### 1. Quantum vs Classical Classification
```bash
python -m src.experiments.runner --config config/experiments/classification_benchmark.yml
```
Compares quantum kernel methods with classical SVM on standard datasets.

#### 2. Variational Quantum Eigensolver (VQE)
```bash
python -m src.experiments.runner --config config/experiments/vqe_optimization.yml
```
Tests VQE for finding ground state energies.

#### 3. Quantum Neural Network Training
```bash
python -m src.experiments.runner --config config/experiments/qnn_training.yml
```
Trains a quantum neural network on a binary classification task.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_quantum_ml/test_circuits.py

# Run tests by marker
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Only integration tests
```

### Code Quality Checks

```bash
# Format code with Black
black src tests --line-length=100

# Lint code
flake8 src tests --max-line-length=100

# Type checking
mypy src --ignore-missing-imports

# Run all checks
black src tests --line-length=100 && \
flake8 src tests --max-line-length=100 && \
pytest --cov=src
```

## 📊 Experiment Guide

### Designing Experiments

1. **Define Research Question**: Clearly state what you're investigating
2. **Choose Datasets**: Select appropriate datasets for your research question
3. **Design Quantum Circuits**: Create quantum circuits that encode your approach
4. **Set Baselines**: Establish classical baselines for comparison
5. **Run Experiments**: Execute experiments with proper controls
6. **Analyze Results**: Statistical analysis and visualization
7. **Document Findings**: Record observations in `docs/experiments/`

### Experiment Best Practices

- **Reproducibility**: Set random seeds (`np.random.seed(42)`)
- **Version Control**: Commit code and configs before long experiments
- **Logging**: Use detailed logging to track experiment progress
- **Resource Management**: Monitor simulator memory usage for large circuits
- **Incremental Testing**: Test on small datasets before scaling up
- **Save Checkpoints**: Regularly save intermediate results
- **Document Everything**: Keep detailed notes in experiment logs

### Analyzing Results

Results are automatically saved to `results/` with:
- **Figures**: Plots and visualizations
- **Metrics**: JSON files with performance metrics
- **Logs**: Detailed execution logs
- **Models**: Trained model parameters

Use the provided Jupyter notebooks in `notebooks/` for interactive analysis and visualization.

## 📚 Documentation

- **Coding Guidelines**: See `.github/copilot-instructions.md` for detailed coding standards
- **Literature Reviews**: Academic papers and references in `docs/literature/`
- **Experiment Logs**: Detailed experiment documentation in `docs/experiments/`
- **API Documentation**: Generated from docstrings using Sphinx (coming soon)

## 🤝 Contributing

This is a research project, but contributions and suggestions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Make your changes following the coding guidelines
4. Run tests and linting (`pytest && black src tests && flake8 src tests`)
5. Commit your changes (`git commit -m 'Add new quantum algorithm'`)
6. Push to the branch (`git push origin feature/new-algorithm`)
7. Open a Pull Request

## 📄 License

This project is part of academic research at Deakin University. Please contact the repository owner for licensing information.

## 👤 Author

**Deakin University Honours Research**
- Institution: Deakin University
- Program: SIT746 - Research Project
- Repository: [bennyp85/SIT746---Research-Project](https://github.com/bennyp85/SIT746---Research-Project)

## 🙏 Acknowledgments

- IBM Quantum team for the Qiskit framework
- Deakin University for research support
- The quantum computing research community

## 📞 Support

For questions or issues:
- Open an issue in this repository
- Contact the project supervisor
- Refer to [Qiskit documentation](https://qiskit.org/documentation/)

## 🔗 Useful Resources

- [Qiskit Textbook](https://qiskit.org/textbook/preface.html) - Learn quantum computing
- [Qiskit Tutorials](https://qiskit.org/documentation/tutorials.html) - Hands-on examples
- [Quantum Machine Learning](https://www.nature.com/articles/nature23474) - Survey paper
- [IBM Quantum Experience](https://quantum-computing.ibm.com/) - Access to real quantum computers

---

**Note**: This is a research project under active development. Code and documentation are continuously evolving as the research progresses.
