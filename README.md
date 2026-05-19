---
title: NeuroPredict
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Psychiatric classification from rs-fMRI connectivity
---

# NeuroPredict

A demo of an end-to-end machine learning pipeline for psychiatric classification from resting-state functional MRI connectivity. Trained on the ABIDE I dataset (n=871, 20 sites), the model uses L1-regularized logistic regression on whole-brain functional connectivity features to distinguish autism from controls.

## What this demo does

- **Pick an example subject** from a small set of pre-loaded ABIDE participants, or **upload your own** 200×200 connectivity matrix.
- See the model's prediction (autism vs. control) with calibrated probability.
- See which **brain connections** drove the prediction, ranked by SHAP attribution.
- See those connections mapped to **canonical resting-state networks** (default mode, salience, frontoparietal, visual, etc.) using the Yeo 7-network parcellation.

## Performance

Mean leave-one-site-out cross-validation AUC: **0.741** across 20 sites. Comparable to published benchmarks on this dataset (Heinsfeld et al. 2017: ~0.70).

## Limitations

This is a demo, not a clinical tool. The model is trained on one disorder, one preprocessing pipeline, and one atlas. Performance varies substantially across sites (AUC range: 0.53–0.93). Predictions on out-of-distribution data should be interpreted cautiously.

## Code and methodology

Full source, training scripts, evaluation pipeline, and a graph neural network comparison: [github.com/ToddHunsaker/neuropredict](https://github.com/ToddHunsaker/neuropredict)
