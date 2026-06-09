"""Extract centroid coordinates for each CC200 region in MNI space.

The CC200 atlas (Craddock 2012) is a 3D NIfTI image where each voxel has an
integer label (0 = background, 1-200 = brain regions). The centroid of a
region is the mean coordinate of all voxels with that label, transformed
into MNI152 space using the image's affine matrix.

Output: app_artifacts/cc200_centroids.npy, shape (200, 3)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from nilearn import datasets, image


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading CC200 atlas...")
    cc200 = datasets.fetch_atlas_craddock_2012()
    cc200_img = image.index_img(cc200["maps"], 19)
    cc200_data = cc200_img.get_fdata().astype(int)
    affine = cc200_img.affine

    n_regions = 200
    centroids = np.full((n_regions, 3), np.nan)

    logger.info("Computing centroids for %d regions...", n_regions)
    for region_id in range(1, n_regions + 1):
        voxel_mask = cc200_data == region_id
        if not voxel_mask.any():
            logger.warning("Region %d has no voxels; leaving centroid as NaN", region_id)
            continue
        # Voxel indices where this region exists
        voxel_indices = np.array(np.where(voxel_mask))  # shape (3, n_voxels)
        # Mean voxel index, then convert to MNI via affine
        mean_voxel = voxel_indices.mean(axis=1)
        mean_voxel_h = np.append(mean_voxel, 1.0)  # homogeneous coords
        mni_coord = affine @ mean_voxel_h
        centroids[region_id - 1] = mni_coord[:3]

    n_valid = int(np.isfinite(centroids).all(axis=1).sum())
    logger.info("Computed %d/%d valid centroids", n_valid, n_regions)

    out_path = Path("app_artifacts/cc200_centroids.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, centroids)
    logger.info("Saved centroids to %s", out_path)


if __name__ == "__main__":
    main()
