# NeuroPredict

Psychiatric classification from resting-state fMRI functional connectivity, with explainable AI and a deployed interactive web demo.

**[Live demo on Hugging Face Spaces](https://huggingface.co/spaces/Saturnalia777/NeuroPredict)**

---

## What it does

NeuroPredict takes a resting-state fMRI functional connectivity matrix -- a 200×200 grid of Pearson correlations between brain region activity time-series -- and predicts whether a subject has autism spectrum disorder or is a neurotypical control. For each prediction, it explains *which brain connections drove the decision* using SHAP attributions, mapped back to anatomical regions and Yeo 7 functional networks, and visualizes them on an interactive 3D brain.

The project spans the full ML pipeline: data loading and preprocessing, classical and graph-based model training, explainability analysis, and a deployed Streamlit app with interactive neuroimaging visualizations.

---

## Demo

The live app lets you pick from 8 bundled example subjects (4 autism, 4 control, drawn from NYU, UM-1, UCLA-1, and PITT acquisition sites) or upload your own 200×200 connectivity matrix. For each subject it shows:

- Predicted diagnosis and autism probability
- Connectivity matrix heatmap
- Interactive 3D brain visualization of the top 15 most influential connections (drag to rotate, scroll to zoom)
- Static four-view glass brain (lateral left/right, top, bottom)
- Per-subject SHAP attribution table with Yeo network labels
- Network involvement bar chart

---

## Results

| Model | Architecture | Evaluation | Mean AUC |
|---|---|---|---|
| Linear baseline | L1-regularized logistic regression | LOSO site-CV (20 sites) | **0.741** |
| BrainGCN | Graph convolutional network | LOSO site-CV (20 sites) | 0.662 |

The linear model outperformed the GNN. This is consistent with findings in the rs-fMRI classification literature: functional connectivity matrices are high-dimensional and noisy, and the inductive bias of GNNs doesn't help when the graph topology itself (which regions connect to which) is the same for every subject. The linear model with L1 regularization selects the most discriminative edges directly. The GNN result is reported honestly as a negative finding.

The linear baseline is competitive with Heinsfeld et al. (2018), which reported AUC ~0.70 on ABIDE using a similar leave-site-out evaluation protocol. This is not 2024 state of the art -- recent deep learning approaches on larger multi-site datasets achieve higher performance -- but it reflects what a well-regularized classical model can do on this benchmark.

---

## Architecture

```
ABIDE rs-fMRI data (1,035 subjects, 20 sites)
        │
        ▼
Craddock CC200 atlas parcellation (200 brain regions)
        │
        ▼
Pearson correlation connectivity matrices (200×200 per subject)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
L1-regularized logistic regression        BrainGCN (PyTorch Geometric)
LOSO site-CV → AUC 0.741                  LOSO site-CV → AUC 0.662
        │                                      │
        ▼                                      ▼
SHAP explainability                       GNNExplainer
(top connections per subject)             (edge attribution aggregation)
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
        Yeo 7-network anatomical mapping
        (CC200 regions → network labels via spatial overlap)
                       │
                       ▼
        Streamlit app (deployed on Hugging Face Spaces)
        - Interactive 3D nilearn connectome widget
        - Static four-view glass brain
        - Per-subject SHAP table and network bar chart
```

---

## Project structure

```
neuropredict/
├── src/
│   ├── neuropredict/
│   │   ├── data.py              # Data loading and preprocessing
│   │   ├── features.py          # Connectivity matrix construction
│   │   ├── models.py            # Linear baseline + LOSO-CV
│   │   ├── gnn.py               # BrainGCN + LOSO-CV-GNN
│   │   ├── explain.py           # SHAP for linear model
│   │   ├── gnn_explain.py       # GNNExplainer + edge attribution
│   │   ├── atlas.py             # CC200 → Yeo 7-network mapping
│   │   └── datasets/
│   │       ├── abide.py         # ABIDE multi-site loader
│   │       └── adhd200_dev.py   # ADHD-200 dev loader (initial exploration)
│   └── streamlit_app.py         # Deployed app entry point
├── scripts/
│   ├── train_baseline.py        # Train and evaluate linear model
│   ├── train_gnn.py             # Train and evaluate BrainGCN
│   ├── run_explain.py           # Compute SHAP values
│   ├── run_gnn_explain.py       # Compute GNN edge attributions
│   ├── compare_explainability.py # Side-by-side SHAP vs GNNExplainer
│   ├── build_app_artifacts.py   # Bundle model + examples for deployment
│   └── extract_centroids.py     # CC200 MNI centroid computation
├── app_artifacts/               # Deployment bundle (model, examples, atlas)
├── tests/                       # Unit tests (12 passing)
├── Dockerfile                   # HF Spaces Docker deployment
├── requirements.txt             # Pinned dependencies for deployment
└── pyproject.toml               # Package configuration
```

---

## Setup

```bash
git clone https://github.com/ToddHunsaker/neuropredict.git
cd neuropredict
conda create -n neuropredict python=3.11
conda activate neuropredict
pip install -e ".[dev]"
```

Download the ABIDE data (nilearn handles this automatically on first run, ~500MB):

```bash
python scripts/inspect_data.py
```

---

## Reproducing the results

```bash
# Train and evaluate the linear baseline (LOSO site-CV, ~5 min)
python scripts/train_baseline.py

# Train and evaluate the GNN (~30-60 min depending on hardware)
python scripts/train_gnn.py

# Compute SHAP attributions for the linear model
python scripts/run_explain.py

# Build the deployment artifact bundle
python scripts/build_app_artifacts.py

# Run the app locally
streamlit run src/streamlit_app.py
```

---

## Key technical decisions

**Why L1 regularization?** The connectivity feature space has ~19,900 dimensions (upper triangle of 200×200) but only ~1,000 subjects. L1 regularization drives most weights to zero, effectively selecting the most discriminative edges. The final model uses ~11,600 non-zero coefficients.

**Why leave-site-out cross-validation?** ABIDE aggregates data from 20 acquisition sites with different scanners, protocols, and demographics. A random train/test split would leak site identity and inflate performance estimates. LOSO-CV trains on 19 sites and tests on the held-out site, forcing the model to generalize across acquisition conditions.

**Why not deploy the GNN?** The GNN underperformed the linear model (AUC 0.662 vs 0.741). Deploying a worse-performing model because it sounds more impressive would be the wrong call. The GNN code and results are in the repository for transparency.

**Why SHAP for a linear model?** A logistic regression is already interpretable via its coefficients -- the weight for each feature is directly interpretable. SHAP was used here to produce per-subject, signed attributions: for each prediction, which connections pushed toward autism and which pushed toward control, by how much. The global model weights tell you what the model learned; the per-subject SHAP values tell you why it made this specific prediction for this specific person.

---

## What I learned

Building this project surfaced several things that aren't obvious from reading papers:

**Site effects dominate.** The biggest signal in ABIDE isn't autism vs. control -- it's which site the data came from. Scanner differences, preprocessing pipelines, and demographic composition all create site-level variance that swamps the diagnostic signal. LOSO-CV is mandatory, not optional.

**GNNs don't automatically win.** The GNN had more parameters and a more sophisticated architecture, and it still lost to logistic regression by a meaningful margin. The inductive bias of graph convolution doesn't help when the graph structure is identical for every subject. For tabular connectivity data, classical methods are a strong baseline worth taking seriously.

**Explainability is the interesting part.** The SHAP analysis revealed that the linear model's top discriminative connections were concentrated in visual and somatomotor networks, while GNNExplainer emphasized default mode and salience networks -- different network fingerprints for the same predictions. Understanding *why* these methods disagree is more scientifically interesting than either result in isolation.

**Deployment is harder than modeling.** The ML pipeline took Phase 1-5. Making it run reliably in a Docker container on Hugging Face Spaces — with correct dependency pinning, Git LFS for binary artifacts, nilearn's atlas download behavior, and cross-browser visualization -- took Phase 6. Both halves matter for a production system.

---

## References

- Craddock et al. (2012). A whole brain fMRI atlas generated via spatially constrained spectral clustering. *Human Brain Mapping*.
- Di Martino et al. (2014). The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*.
- Heinsfeld et al. (2018). Identification of autism spectrum disorder using deep learning and the ABIDE dataset. *NeuroImage: Clinical*.
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Yeo et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology*.

---

## License

MIT
