# Experiment Logs

This directory contains detailed logs and analysis of experiments conducted during the research project.

## Organization

Create a directory for each major experiment:
```
experiments/
├── experiment_001_baseline/
│   ├── README.md          # Experiment description and results
│   ├── config.yml         # Configuration used
│   ├── analysis.ipynb     # Jupyter notebook for analysis
│   └── notes.md           # Observations and notes
├── experiment_002_vqe/
│   └── ...
```

## Experiment Template

Each experiment directory should include:

1. **README.md**: Overview, hypothesis, methodology, results, conclusions
2. **config.yml**: Exact configuration parameters used
3. **analysis.ipynb**: Data analysis and visualization
4. **notes.md**: Running notes, observations, issues encountered

## Naming Convention

Use the format: `experiment_XXX_description`
- XXX: Sequential number (001, 002, etc.)
- description: Brief descriptive name (e.g., baseline, vqe_h2, qnn_iris)

## Best Practices

- Document as you go, not after the fact
- Include failed experiments for learning
- Note any deviations from the plan
- Record computational resources used
- Save all plots and figures with descriptive names
