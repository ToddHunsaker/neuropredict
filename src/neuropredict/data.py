"""Top-level data router.

This module is the single entry point downstream code uses to load any
dataset. It delegates to the appropriate module in `neuropredict.datasets`
based on the `dataset` argument.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from neuropredict.datasets.base import Dataset

logger = logging.getLogger(__name__)

# Map dataset names to loader modules. Keeps the dispatch logic in one place.
_LOADERS = {
    "abide": "neuropredict.datasets.abide",
    "adhd200_dev": "neuropredict.datasets.adhd200_dev",
    "adhd200_full": "neuropredict.datasets.adhd200_full",
}


def load_dataset(dataset: str, **kwargs) -> Dataset:
    """Load a dataset by name.

    Parameters
    ----------
    dataset
        One of: 'abide', 'adhd200_dev', 'adhd200_full'.
    **kwargs
        Passed through to the dataset-specific loader.

    Returns
    -------
    A `Dataset` instance.
    """
    if dataset not in _LOADERS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {sorted(_LOADERS)}"
        )

    import importlib

    module = importlib.import_module(_LOADERS[dataset])
    return module.load(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache a dataset")
    parser.add_argument(
        "--dataset",
        choices=sorted(_LOADERS),
        required=True,
        help="Which dataset to load",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Limit number of subjects (for fast iteration). Default: all available.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Where to cache raw downloads",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save the cached Dataset .npz "
        "(default: data/processed/<dataset>.npz)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    kwargs = {}
    if args.n_subjects is not None:
        kwargs["n_subjects"] = args.n_subjects
    if args.data_dir is not None:
        kwargs["data_dir"] = args.data_dir

    ds = load_dataset(args.dataset, **kwargs)

    output_path = (
        Path(args.output)
        if args.output
        else Path("data/processed") / f"{ds.name}.npz"
    )
    ds.save(output_path)
    logger.info(
        "Saved %s (%d subjects, %d regions) to %s",
        ds.name,
        ds.n_subjects,
        ds.n_regions,
        output_path,
    )


if __name__ == "__main__":
    main()
