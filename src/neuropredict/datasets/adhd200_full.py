"""Full ADHD-200 dataset (~973 subjects) from PCP S3.

NOT YET IMPLEMENTED. The PCP S3 layout for ADHD-200 (under
`data/Projects/ADHD200/Outputs/cpac/raw_outputs/`) is a raw CPAC dump
rather than a curated derivatives release, which makes the loader
considerably more involved than the ABIDE equivalent. Tracking issue:
https://github.com/ToddHunsaker/neuropredict/issues (to be filed).

Planned approach:
1. Enumerate subject directories via paginated S3 list-bucket calls.
2. For each subject, fetch ROI time series from
   `<subject>/roi_timeseries/_scan_rest_1/<atlas-specific-path>`.
3. Pull phenotypic data from `data/Projects/ADHD200/Resources/`.
4. Filter to subjects passing quality checks.
5. Compute connectivity matrices and return a Dataset.

In the meantime, use `adhd200_dev` for small-scale ADHD experiments
(40 subjects, via nilearn) or `abide` for full-scale rs-fMRI
classification work.
"""

from __future__ import annotations

from pathlib import Path

from neuropredict.datasets.base import Dataset


def load(
    data_dir: str | Path | None = None,  # noqa: ARG001
    atlas: str = "cc200",  # noqa: ARG001
    **kwargs,  # noqa: ARG001
) -> Dataset:
    """Load the full ADHD-200 dataset. Not yet implemented."""
    raise NotImplementedError(
        "Full ADHD-200 loader is not yet implemented. "
        "See module docstring for the planned approach. "
        "Use `neuropredict.datasets.adhd200_dev` (40 subjects via nilearn) or "
        "`neuropredict.datasets.abide` (full ABIDE) in the meantime."
    )
