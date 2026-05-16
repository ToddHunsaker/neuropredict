"""Quick inspection of the development subset.

Run after `python -m neuropredict.data --fetch-small --compute-connectivity`
to sanity-check the connectivity matrices.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(npz_path: str = "data/processed/dev_subset.npz") -> None:
    path = Path(npz_path)
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run:\n  "
            "python -m neuropredict.data --fetch-small --compute-connectivity"
        )

    data = np.load(path, allow_pickle=True)
    conn = data["connectivity"]
    labels = data["labels"]
    diagnosis = data["diagnosis"]

    print(f"Connectivity shape: {conn.shape}")  # (n_subjects, n_regions, n_regions)
    print(f"Regions: {len(labels)}")
    print(f"Class balance: {(diagnosis == 0).sum()} controls, {(diagnosis == 1).sum()} ADHD")
    print(f"Mean off-diagonal correlation: {_mean_offdiag(conn):.3f}")
    print(f"Range: [{conn.min():.3f}, {conn.max():.3f}]")

    # Group-mean matrices, side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label, dx in zip(axes, ["Controls", "ADHD"], [0, 1]):
        mean_mat = conn[diagnosis == dx].mean(axis=0)
        np.fill_diagonal(mean_mat, 0)
        im = ax.imshow(mean_mat, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        ax.set_title(f"{label} (n={int((diagnosis == dx).sum())})")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    out = Path("results") / "group_mean_connectivity.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nSaved group-mean figure to {out}")


def _mean_offdiag(conn: np.ndarray) -> float:
    n = conn.shape[1]
    mask = ~np.eye(n, dtype=bool)
    return float(conn[:, mask].mean())


if __name__ == "__main__":
    main()
