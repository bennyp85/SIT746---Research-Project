# Quick Start Guide: SAX Trend Prediction Notebook

## Purpose
This notebook compares quantum machine learning methods (QSVC, VQC) with classical RBF-SVC for trend prediction on SAX-encoded time series data.

## Configuration (Cell 2)
All parameters are fixed for reproducibility:
```python
EXPERIMENT_PARAMS = {
    "sax_window": 10,           # SAX window size (FIXED)
    "sax_alphabet": 3,          # Number of symbols (FIXED)
    "training_fraction": 0.8,   # 80% train, 20% test
    "repetitions_quantum": 1,   # Quantum circuit repetitions (FIXED)
    "timeseries_points": 320,   # Time series length
    "gaussian_noise": 0.10,     # Noise level
    "trend_delta": 0.04,        # Threshold for trend classification
    "data_seed": 123,           # Random seed for data
    "quantum_seed": 12345,      # Random seed for quantum algorithms
}
```

## How to Run

### 1. Install Dependencies
```bash
pip install numpy matplotlib scikit-learn pyts qiskit qiskit-machine-learning
```

### 2. Execute Notebook
Run all cells in order (Kernel → Restart & Run All)

The notebook will:
1. Generate synthetic noisy sinusoidal time series
2. Apply SAX encoding with fixed parameters
3. Map SAX symbols to angles using simple linear spacing
4. Train three models: RBF-SVC, VQC, QSVC
5. Generate comparative results table
6. Create thesis-ready visualizations

## Expected Outputs

### Section 9: Results Summary Table
```
Model         Accuracy    Macro F1    Calib. Error    Kernel Rank
RBF-SVC       X.XXX       X.XXX       X.XXX           N/A
VQC           X.XXX       X.XXX       N/A             N/A
QSVC          X.XXX       X.XXX       N/A             XX
```

### Section 10: Visualizations
1. **Confusion Matrices** (3 plots)
   - RBF-SVC
   - VQC
   - QSVC
   
2. **Calibration Reliability Diagram** (1 plot)
   - RBF-SVC only (requires probability estimates)
   
3. **Kernel Eigenvalue Spectrum** (1 plot)
   - QSVC only (shows effective rank)

## Interpreting Results

### Accuracy & F1-Score
- **Accuracy:** Overall correctness
- **Macro F1:** Average F1 across classes (better for imbalanced data)
- **Baseline:** Random guessing ≈ 0.333 (3 classes), Majority class ≈ 0.4-0.5

### Calibration Error
- **< 0.05:** Well-calibrated (confidence matches accuracy)
- **0.05-0.10:** Moderately calibrated
- **> 0.10:** Poorly calibrated (overconfident or underconfident)

### Effective Rank
- **High rank (close to total):** Kernel uses full feature space
- **Low rank:** Kernel projects to lower-dimensional subspace
- **Interpretation:** Higher isn't always better - indicates complexity vs. simplicity trade-off

### Confusion Matrix
- **Diagonal:** Correct predictions
- **Off-diagonal:** Misclassifications
- **Look for:** Class collapse (all predictions in one column)

## Troubleshooting

### ImportError: No module named 'qiskit'
```bash
pip install qiskit qiskit-machine-learning
```

### DeprecationWarning from Qiskit
These are expected with rapidly evolving Qiskit versions. The code uses current stable APIs.

### Very long execution time
- VQC training can take 5-15 minutes depending on CPU
- QSVC kernel computation can take 2-5 minutes
- Total runtime: 10-25 minutes for full notebook

### Memory errors
- Reduce `timeseries_points` from 320 to 160
- This will reduce train/test samples but maintain structure

## Thesis Usage

### For Results Chapter
1. Run notebook to generate all outputs
2. Export visualizations: Right-click plot → Save Image As...
3. Copy summary table to LaTeX/Word
4. Report all metrics: Accuracy, F1, Calibration Error, Effective Rank

### For Discussion
- **SAX encoding:** Mention fixed window=10, alphabet=3
- **Angle mapping:** Cite simple linear spacing rationale
- **Performance:** Compare quantum vs classical
- **Limitations:** Small dataset, synthetic data, fixed configuration

### For Reproducibility
- Include `EXPERIMENT_PARAMS` in appendix
- Note: Results are deterministic given fixed seeds
- Python version, package versions matter (record in thesis)

## Customization (Advanced)

### Change Time Series
Modify `make_dataset_abc()` in Cell 4:
- Different signal: Replace `np.sin(t)` with your function
- Real data: Load from file instead of synthetic generation

### Adjust Trend Threshold
Modify `EXPERIMENT_PARAMS["trend_delta"]`:
- Smaller value: More sensitive trend detection
- Larger value: Only detect significant trends

### Add More Models
Follow pattern in existing model cells:
1. Define `train_[model]()` function
2. Return dictionary with: accuracy, f1_macro, predictions
3. Add to main execution cell (Cell 14)
4. Add to visualization cell (Cell 16) if needed

## File Structure
```
sax_kernel_svc._2.ipynb          # Main notebook (THIS FILE)
REFACTORING_NOTES.md             # Detailed refactoring documentation
SAX_NOTEBOOK_GUIDE.md            # This guide
```

## Support
For issues with:
- **Notebook execution:** Check dependencies and Qiskit version
- **Understanding outputs:** See "Interpreting Results" section
- **Thesis integration:** See "Thesis Usage" section
- **Code modifications:** See REFACTORING_NOTES.md for structure details

## Version Info
- **Notebook Version:** Refactored (2024)
- **Qiskit:** 0.40+ recommended
- **Python:** 3.9+ required
- **Last Updated:** See git commit history
