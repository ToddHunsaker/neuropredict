# NeuroPredict

ADHD classification from resting-state fMRI functional connectivity, with explainable AI and a deployed web interface.

## Overview

End-to-end ML pipeline that:
1. Loads preprocessed connectivity data from the ADHD-200 dataset
2. Trains and compares classical ML and graph neural network models
3. Provides SHAP-based explanations mapped to brain networks
4. Serves predictions via a Streamlit app deployed on Hugging Face Spaces

## Project structure

```
neuropredict/
├── src/neuropredict/    # Core package
│   ├── data.py          # Data loading and preprocessing
│   ├── features.py      # Connectivity matrix construction
│   ├── models.py        # Classical ML models
│   ├── gnn.py           # Graph neural network
│   ├── explain.py       # SHAP and GNN explainability
│   └── viz.py           # Brain visualization utilities
├── notebooks/           # Exploratory analysis
├── tests/               # Unit tests
├── configs/             # Experiment configurations
├── data/                # Datasets (gitignored)
├── models/              # Trained model artifacts
├── results/             # Figures, metrics, reports
├── app.py               # Streamlit application
└── pyproject.toml       # Package configuration
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick start

```bash
# Fetch a small ADHD-200 subset (40 subjects) for development
python -m neuropredict.data --fetch-small

# Inspect what we loaded
python scripts/inspect_data.py
```

## Status

Phase 1: Data acquisition and exploration — in progress.
