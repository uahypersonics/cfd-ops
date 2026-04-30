"""cfd-ops: Operations for transforming, extracting, remapping, and combining CFD datasets."""

from importlib.metadata import version

from cfd_ops.operations import (
    cut_dataset,
    extend_dataset,
    merge_datasets,
    merge_files_to_hdf5,
    rotate_dataset,
    translate_dataset,
    write_merged_hdf5,
)

__all__ = [
    "cut_dataset",
    "extend_dataset",
    "merge_datasets",
    "merge_files_to_hdf5",
    "rotate_dataset",
    "translate_dataset",
    "write_merged_hdf5",
    "__version__",
]

__version__ = version("cfd-ops")
