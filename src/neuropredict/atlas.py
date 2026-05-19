"""Map CC200 atlas regions to Yeo 7-network parcellation.

The CC200 atlas (Craddock 2012) defines 200 functional parcels, but doesn't
ship with network membership labels. The Yeo 7-network atlas (Yeo et al.
2011) is the field-standard resting-state network parcellation. We compute
the spatial overlap between each CC200 region and each Yeo network, then
assign each region to its dominant network.

Yeo network IDs:
    0 = (background / outside cortex)
    1 = Visual
    2 = Somatomotor
    3 = Dorsal Attention
    4 = Ventral Attention / Salience
    5 = Limbic
    6 = Frontoparietal / Control
    7 = Default Mode
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nilearn import datasets, image

logger = logging.getLogger(__name__)

YEO_NETWORK_NAMES = {
    0: "background",
    1: "visual",
    2: "somatomotor",
    3: "dorsal_attention",
    4: "salience",
    5: "limbic",
    6: "frontoparietal",
    7: "default_mode",
}


@dataclass
class RegionAnnotation:
    """One CC200 region's network membership and overlap statistics."""
    region_idx: int          # CC200 region index (0-199)
    dominant_network_id: int  # 0-7
    dominant_network_name: str
    overlap_fraction: float  # what fraction of region voxels fall in dominant net
    n_voxels: int            # size of the region in atlas voxels


def compute_cc200_to_yeo_mapping(
    cache_path: str | Path | None = None,
) -> list[RegionAnnotation]:
    """For each CC200 region, find its dominant Yeo network by spatial overlap.

    Optionally cache the result to a JSON file. If the cache exists, just load it.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached CC200->Yeo mapping from %s", cache_path)
            with cache_path.open() as fh:
                cached = json.load(fh)
            return [RegionAnnotation(**entry) for entry in cached]

    logger.info("Fetching CC200 (Craddock 2012) atlas...")
    cc200 = datasets.fetch_atlas_craddock_2012()
    # Newer nilearn exposes a single 'maps' key; index 19 is the 200-region scorr_mean
    cc200_img = image.index_img(cc200["maps"], 19)

    logger.info("Fetching Yeo 7-network atlas...")
    yeo = datasets.fetch_atlas_yeo_2011()
    yeo_img = image.load_img(yeo["maps"])
    # Yeo atlas may have an extra singleton dim; squeeze if needed
    yeo_data = yeo_img.get_fdata().squeeze()

    # Resample CC200 to match Yeo's voxel grid so we can compare voxel-wise
    logger.info("Resampling CC200 to Yeo space...")
    cc200_resampled = image.resample_to_img(
        cc200_img, yeo_img, interpolation="nearest"
    )
    cc200_data = cc200_resampled.get_fdata().astype(int)

    annotations = []
    unique_regions = sorted(int(r) for r in np.unique(cc200_data) if r > 0)
    logger.info("Computing overlap for %d CC200 regions...", len(unique_regions))

    for region_id in unique_regions:
        region_mask = cc200_data == region_id
        n_voxels = int(region_mask.sum())
        if n_voxels == 0:
            continue

        yeo_labels_here = yeo_data[region_mask].astype(int)
        # Count each Yeo network's presence in this region (excluding background=0)
        counts = np.bincount(yeo_labels_here, minlength=8)
        cortical_counts = counts[1:]  # drop background
        if cortical_counts.sum() == 0:
            dominant_id = 0
            overlap = 0.0
        else:
            dominant_id = int(np.argmax(cortical_counts)) + 1  # +1 to undo the slice
            overlap = float(cortical_counts.max() / n_voxels)

        annotations.append(
            RegionAnnotation(
                region_idx=region_id - 1,  # CC200 is 1-indexed; we use 0-indexed
                dominant_network_id=dominant_id,
                dominant_network_name=YEO_NETWORK_NAMES[dominant_id],
                overlap_fraction=overlap,
                n_voxels=n_voxels,
            )
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as fh:
            json.dump([a.__dict__ for a in annotations], fh, indent=2)
        logger.info("Cached CC200->Yeo mapping to %s", cache_path)

    return annotations


def summarize_top_features_by_network(
    top_features: list,
    region_to_network: dict[int, str],
) -> dict[str, int]:
    """Count how many top connections involve each Yeo network.

    A connection counts toward a network if either endpoint is in that network.
    """
    counts: dict[str, int] = {name: 0 for name in YEO_NETWORK_NAMES.values()}
    for f in top_features:
        net_a = region_to_network.get(f.region_a, "background")
        net_b = region_to_network.get(f.region_b, "background")
        counts[net_a] += 1
        if net_b != net_a:
            counts[net_b] += 1
    return counts
