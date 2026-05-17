"""Dataset loaders for psychiatric classification from rs-fMRI connectivity.

Each module in this package implements a `load(...)` function returning a
`Dataset` instance (see `base.py`). Downstream pipeline code reads from the
Dataset interface and stays dataset-agnostic.

Currently supported:
- abide : ABIDE PCP (autism, ~1100 subjects) — primary working dataset
- adhd200_dev : Nilearn's 40-subject ADHD-200 subset — small, useful for tests
- adhd200_full : Full ADHD-200 from PCP S3 — not yet implemented
"""

from neuropredict.datasets.base import Dataset

__all__ = ["Dataset"]
