"""Streamlit app for the NeuroPredict project.

Loads a pretrained L1 logistic regression model that classifies subjects
as autism vs. control from resting-state fMRI functional connectivity,
and offers two ways to try it:

1. Pick from a set of bundled example ABIDE subjects
2. Upload your own 200x200 connectivity matrix (.npy or .csv)

The app then displays the model's prediction, per-subject SHAP attributions
for the most influential connections, and a network-level summary mapped
to Yeo's 7 canonical resting-state networks.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from nilearn import plotting

ARTIFACTS_DIR = Path("app_artifacts")
N_REGIONS_EXPECTED = 200


@st.cache_resource
def load_model_and_scaler():
    bundle = joblib.load(ARTIFACTS_DIR / "linear_model.joblib")
    return bundle["model"], bundle["scaler"], bundle["C"]


@st.cache_resource
def load_atlas():
    with (ARTIFACTS_DIR / "atlas_mapping.json").open() as fh:
        data = json.load(fh)
    region_to_network = {int(k): v for k, v in data["region_to_network"].items()}
    return region_to_network, data["n_regions"]


@st.cache_resource
def load_examples():
    matrices = np.load(ARTIFACTS_DIR / "example_connectivity.npy")
    with (ARTIFACTS_DIR / "examples.json").open() as fh:
        meta = json.load(fh)
    return matrices, meta["examples"]


@st.cache_resource
def load_dataset_meta():
    with (ARTIFACTS_DIR / "dataset_meta.json").open() as fh:
        return json.load(fh)

@st.cache_resource
def load_centroids():
    return np.load(ARTIFACTS_DIR / "cc200_centroids.npy")


@st.cache_resource
def get_shap_explainer(_model, _scaler, _examples_matrices):
    """Build a SHAP explainer using the example subjects as the background."""
    iu = np.triu_indices(N_REGIONS_EXPECTED, k=1)
    background_features = np.empty((_examples_matrices.shape[0], len(iu[0])))
    for i, mat in enumerate(_examples_matrices):
        background_features[i] = mat[iu]
    background_features = np.arctanh(np.clip(background_features, -0.999, 0.999))
    background_scaled = _scaler.transform(background_features)
    return shap.LinearExplainer(_model, background_scaled)


def connectivity_to_feature_vector(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle features and apply Fisher z-transform."""
    iu = np.triu_indices(matrix.shape[0], k=1)
    raw = matrix[iu]
    return np.arctanh(np.clip(raw, -0.999, 0.999))


def predict_and_explain(matrix, model, scaler, explainer):
    features = connectivity_to_feature_vector(matrix).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prob_autism = float(model.predict_proba(features_scaled)[0, 1])
    prediction = "autism" if prob_autism >= 0.5 else "control"
    shap_values = explainer.shap_values(features_scaled)[0]
    return prob_autism, prediction, shap_values


def top_k_features(shap_values, n_regions, k=15):
    iu = np.triu_indices(n_regions, k=1)
    mag = np.abs(shap_values)
    top_idx = np.argsort(mag)[::-1][:k]
    return [
        {
            "rank": rank + 1,
            "region_a": int(iu[0][i]),
            "region_b": int(iu[1][i]),
            "shap": float(shap_values[i]),
            "abs_shap": float(mag[i]),
        }
        for rank, i in enumerate(top_idx)
    ]


def parse_uploaded_matrix(uploaded_file) -> np.ndarray | None:
    """Accept .npy or .csv and return a numpy array. Validate dimensions."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".npy"):
            arr = np.load(uploaded_file)
        elif name.endswith(".csv"):
            arr = pd.read_csv(uploaded_file, header=None).to_numpy()
        else:
            st.error("Unsupported file type. Please upload a .npy or .csv file.")
            return None
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        st.error(f"Matrix must be square. Got shape {arr.shape}.")
        return None
    if arr.shape[0] != N_REGIONS_EXPECTED:
        st.error(
            f"Matrix must be {N_REGIONS_EXPECTED}x{N_REGIONS_EXPECTED}. "
            f"Got {arr.shape}."
        )
        return None
    if not np.allclose(arr, arr.T, atol=1e-4):
        st.warning("Matrix is not symmetric. Symmetrizing automatically.")
        arr = (arr + arr.T) / 2
    return arr


def main():
    st.set_page_config(
        page_title="NeuroPredict",
        page_icon=":brain:",
        layout="wide",
    )

    model, scaler, c_param = load_model_and_scaler()
    region_to_network, n_regions = load_atlas()
    example_matrices, example_meta = load_examples()
    dataset_meta = load_dataset_meta()
    explainer = get_shap_explainer(model, scaler, example_matrices)

    st.title("NeuroPredict")
    st.markdown(
        "**A psychiatric classification demo from resting-state fMRI connectivity.** "
        "Pick an example subject or upload your own functional connectivity matrix to "
        "see the model's autism vs. control prediction and which connections drove it."
    )

    with st.expander("About this project"):
        st.markdown(
            f"""
            This is a portfolio project demonstrating end-to-end machine learning for
            psychiatric neuroimaging. The model is an L1-regularized logistic regression
            trained on **{dataset_meta['n_subjects']} subjects** from the public
            **ABIDE I** dataset (Autism Brain Imaging Data Exchange), using
            functional connectivity from the **CC200 atlas** (200 brain regions, so
            {dataset_meta['n_features']:,} connection features).

            **Performance** (leave-one-site-out cross-validation, 20 sites):
            mean accuracy 0.665, mean AUC 0.741 — consistent with published benchmarks.

            **Repository:** [github.com/ToddHunsaker/neuropredict](https://github.com/ToddHunsaker/neuropredict)
            """
        )

    st.sidebar.header("Choose an input")
    mode = st.sidebar.radio(
        "Mode",
        options=["Pick an example subject", "Upload your own matrix"],
        index=0,
    )

    matrix = None
    subject_info = None

    if mode == "Pick an example subject":
        options = [
            f"#{e['subject_idx']} | {e['site']} | {e['diagnosis_name']} | "
            f"age {e['age']:.1f} | sex {'M' if e['sex'] == 1 else 'F'}"
            for e in example_meta
        ]
        choice = st.sidebar.selectbox("Example subjects", options)
        idx = options.index(choice)
        matrix = example_matrices[idx]
        subject_info = example_meta[idx]
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload a 200x200 connectivity matrix",
            type=["npy", "csv"],
            help="Pearson correlation matrix from the CC200 atlas. "
            "Values should be in [-1, 1].",
        )
        if uploaded is not None:
            matrix = parse_uploaded_matrix(uploaded)

    if matrix is None:
        st.info("Pick an example or upload a matrix from the sidebar to see results.")
        return

    prob_autism, prediction, shap_values = predict_and_explain(
        matrix, model, scaler, explainer
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Functional connectivity matrix")
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xlabel("Brain region")
        ax.set_ylabel("Brain region")
        plt.colorbar(im, ax=ax, label="Pearson r")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Model prediction")
        confidence = max(prob_autism, 1 - prob_autism)
        st.metric("Predicted class", prediction)
        st.metric("Probability of autism", f"{prob_autism:.3f}")
        st.metric("Confidence", f"{confidence:.3f}")
        st.progress(prob_autism)

        if subject_info is not None:
            st.markdown("**Ground truth (this example)**")
            st.write(f"- Diagnosis: `{subject_info['diagnosis_name']}`")
            st.write(f"- Site: `{subject_info['site']}`")
            correct = (subject_info["diagnosis"] == 1) == (prediction == "autism")
            st.write(f"- Model correct: `{correct}`")

    # Build the connection data once; both 3D and 4-pane use it
    top_features = top_k_features(shap_values, n_regions, k=15)
    centroids = load_centroids()

    adjacency = np.zeros((n_regions, n_regions))
    for f in top_features:
        a, b = f["region_a"], f["region_b"]
        adjacency[a, b] = f["shap"]
        adjacency[b, a] = f["shap"]

    valid_mask = np.isfinite(centroids).all(axis=1)
    valid_indices = np.where(valid_mask)[0]
    adjacency_valid = adjacency[np.ix_(valid_indices, valid_indices)]
    centroids_valid = centroids[valid_indices]

    st.subheader("Brain visualization: interactive 3D")
    st.caption(
        "Drag to rotate, scroll to zoom. "
        "Red lines push the prediction toward autism; blue toward control."
    )
    view = plotting.view_connectome(
        adjacency_valid,
        centroids_valid,
        edge_threshold=None,
        edge_cmap="RdBu_r",
        symmetric_cmap=True,
        linewidth=4.0,
        node_size=4.0,
        colorbar=True,
    )
    st.components.v1.html(view.get_iframe(), height=520, scrolling=False)

    st.subheader("Brain visualization: static four-view")
    st.caption(
        "Lateral left, lateral right, top, and bottom views. "
        "Same connections as the 3D view above."
    )
    edge_max = max(abs(adjacency.min()), abs(adjacency.max()))
    fig_brain = plotting.plot_connectome(
        adjacency_valid,
        centroids_valid,
        edge_threshold=None,
        node_color="lightgray",
        node_size=20,
        edge_kwargs={"linewidth": 2},
        edge_cmap="RdBu_r",
        edge_vmin=-edge_max,
        edge_vmax=edge_max,
        display_mode="lyrz",
        figure=plt.figure(figsize=(12, 4)),
    )
    st.pyplot(fig_brain.frame_axes.figure)
    plt.close("all")

    st.subheader("Most influential connections (per-subject SHAP)")
    rows = []
    for f in top_features:
        net_a = region_to_network.get(f["region_a"], "unknown")
        net_b = region_to_network.get(f["region_b"], "unknown")
        rows.append({
            "rank": f["rank"],
            "region_a": f["region_a"],
            "region_b": f["region_b"],
            "network_a": net_a,
            "network_b": net_b,
            "SHAP": f["shap"],
            "abs_SHAP": f["abs_shap"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Networks involved (top 15 connections)")
    network_counts = {}
    for f in top_features:
        net_a = region_to_network.get(f["region_a"], "unknown")
        net_b = region_to_network.get(f["region_b"], "unknown")
        network_counts[net_a] = network_counts.get(net_a, 0) + 1
        if net_b != net_a:
            network_counts[net_b] = network_counts.get(net_b, 0) + 1

    names = sorted(network_counts.keys())
    counts = [network_counts[n] for n in names]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(names, counts, color="steelblue", alpha=0.85)
    ax2.set_ylabel("Endpoints in top 15 connections")
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels(names, rotation=30, ha="right")
    for i, c in enumerate(counts):
        ax2.text(i, c + 0.1, str(c), ha="center", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    with st.expander("Download this result"):
        result = {
            "prediction": prediction,
            "prob_autism": prob_autism,
            "confidence": confidence,
            "top_features": top_features,
            "network_counts": network_counts,
        }
        buf = io.BytesIO(json.dumps(result, indent=2).encode("utf-8"))
        st.download_button(
            "Download JSON",
            data=buf,
            file_name="neuropredict_result.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
