"""Merge operations for assembling multi-timestep outputs."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from cfd_io import Dataset, Field, read_file

from cfd_ops.operations.common import as_field, copy_attrs, require_structured_grid


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _flow_arrays(flow: dict[str, Field | np.ndarray]) -> dict[str, np.ndarray]:
    """Convert flow mapping into plain ndarray dictionary."""
    # build output container
    out: dict[str, np.ndarray] = {}

    # convert each flow variable to ndarray copy
    for name, value in flow.items():
        field_obj = as_field(value)
        out[name] = np.array(field_obj.data, copy=True)

    return out


def _validate_same_grid(
    ref: Dataset,
    other: Dataset,
    *,
    strict_grid: bool,
    rtol: float,
    atol: float,
) -> None:
    """Validate grid compatibility between two datasets."""
    # validate grid types
    ref_grid = require_structured_grid(ref, "merge_datasets")
    other_grid = require_structured_grid(other, "merge_datasets")

    # check shape equality first
    if ref_grid.x.shape != other_grid.x.shape:
        raise ValueError(
            "merge grid mismatch: shapes differ "
            f"{ref_grid.x.shape} vs {other_grid.x.shape}"
        )

    # optionally enforce coordinate equality within tolerance
    if strict_grid:
        if not np.allclose(ref_grid.x, other_grid.x, rtol=rtol, atol=atol):
            raise ValueError("merge grid mismatch: x coordinates differ")
        if not np.allclose(ref_grid.y, other_grid.y, rtol=rtol, atol=atol):
            raise ValueError("merge grid mismatch: y coordinates differ")
        if not np.allclose(ref_grid.z, other_grid.z, rtol=rtol, atol=atol):
            raise ValueError("merge grid mismatch: z coordinates differ")


# --------------------------------------------------
# public API
# --------------------------------------------------
def merge_datasets(
    datasets: list[Dataset],
    *,
    timestep_ids: list[int] | None = None,
    strict_grid: bool = True,
    grid_rtol: float = 1.0e-10,
    grid_atol: float = 1.0e-12,
) -> tuple[Dataset, dict[int, dict[str, np.ndarray]]]:
    """Merge multiple single-snapshot datasets into timestep-indexed flow blocks.

    Args:
        datasets: Input datasets to merge in order.
        timestep_ids: Optional explicit timestep ids for output groups.
        strict_grid: If True, coordinates must match within tolerance.
        grid_rtol: Relative tolerance for strict grid checks.
        grid_atol: Absolute tolerance for strict grid checks.

    Returns:
        Tuple of (base_dataset, timestep_flow) where base_dataset carries
        reference grid and merged attrs, and timestep_flow maps timestep id to
        flow-variable ndarray dictionaries.

    Raises:
        ValueError: If no datasets are provided or compatibility fails.
    """
    # validate inputs
    if not datasets:
        raise ValueError("merge_datasets requires at least one dataset")

    # create default timestep ids if not provided
    if timestep_ids is None:
        timestep_ids = list(range(1, len(datasets) + 1))

    # validate timestep id cardinality
    if len(timestep_ids) != len(datasets):
        raise ValueError("timestep_ids length must match number of datasets")

    # capture reference dataset and grid
    ref = datasets[0]
    ref_grid = require_structured_grid(ref, "merge_datasets")

    # merge attrs from first dataset and annotate merge details
    attrs_out = copy_attrs(ref.attrs)
    attrs_out["cfd_ops_merge_count"] = len(datasets)
    attrs_out["timesteps"] = list(int(v) for v in timestep_ids)

    # collect timestep flow blocks
    timestep_flow: dict[int, dict[str, np.ndarray]] = {}

    # validate each dataset and convert flow to ndarray dict
    for idx, (dataset, timestep_id) in enumerate(zip(datasets, timestep_ids), start=1):
        _validate_same_grid(
            ref,
            dataset,
            strict_grid=strict_grid,
            rtol=grid_rtol,
            atol=grid_atol,
        )

        ts_flow = _flow_arrays(dataset.flow)
        ts_flow["_iteration"] = int(timestep_id)
        ts_flow["_source_index"] = int(idx)
        timestep_flow[int(timestep_id)] = ts_flow

    # build base dataset carrying merged attrs and reference grid
    base_dataset = Dataset(
        grid=ref_grid,
        flow=ref.flow,
        attrs=attrs_out,
    )

    return base_dataset, timestep_flow


def write_merged_hdf5(
    output_path: str | Path,
    *,
    base_dataset: Dataset,
    timestep_flow: dict[int, dict[str, np.ndarray]],
    dtype: str = "f",
) -> Path:
    """Write merged timestep flow blocks to grouped HDF5 output.

    Args:
        output_path: Output HDF5 path.
        base_dataset: Dataset containing grid and root attrs.
        timestep_flow: Mapping of timestep id to flow arrays.
        dtype: NumPy dtype string for stored arrays.

    Returns:
        Path to output file.

    Raises:
        TypeError: If base_dataset grid is not structured.
        ValueError: If timestep_flow is empty.
    """
    # validate inputs
    grid = require_structured_grid(base_dataset, "write_merged_hdf5")

    if not timestep_flow:
        raise ValueError("write_merged_hdf5 requires at least one timestep")

    # convert output path to Path object
    out_path = Path(output_path)

    # write grouped HDF5 structure
    with h5py.File(out_path, "w") as fobj:
        # write grid group once
        grid_grp = fobj.create_group("grid")
        grid_grp.create_dataset("x", data=grid.x, dtype=dtype)
        grid_grp.create_dataset("y", data=grid.y, dtype=dtype)
        grid_grp.create_dataset("z", data=grid.z, dtype=dtype)

        # write flow groups for each timestep
        flow_grp = fobj.create_group("flow")
        for timestep_id in sorted(timestep_flow):
            ts_name = f"{int(timestep_id):05d}"
            ts_grp = flow_grp.create_group(ts_name)
            ts_data = timestep_flow[timestep_id]

            # write variables and reserve underscore keys for metadata
            for name, data in ts_data.items():
                if name.startswith("_"):
                    continue
                ts_grp.create_dataset(name, data=data, dtype=dtype)

            # write optional timestep metadata
            if "_iteration" in ts_data:
                ts_grp.attrs["iteration"] = int(ts_data["_iteration"])
            if "_solution_time" in ts_data:
                ts_grp.attrs["solution_time"] = float(ts_data["_solution_time"])
            if "_source_index" in ts_data:
                ts_grp.attrs["source_index"] = int(ts_data["_source_index"])

        # write root attributes
        for key, value in (base_dataset.attrs or {}).items():
            if value is None:
                continue
            fobj.attrs[key] = value

    return out_path


def merge_files_to_hdf5(
    *,
    input_files: list[str | Path],
    output_file: str | Path,
    input_grids: list[str | Path | None] | None = None,
    input_iterations: list[int] | None = None,
    strict_grid: bool = True,
    grid_rtol: float = 1.0e-10,
    grid_atol: float = 1.0e-12,
    dtype: str = "f",
) -> Path:
    """Read multiple source files and write a merged multi-timestep HDF5 file.

    Args:
        input_files: Input flow files in merge order.
        output_file: Output HDF5 path.
        input_grids: Optional companion grid paths for split formats.
        input_iterations: Optional timestep index per input.
        strict_grid: If True, coordinates must match within tolerance.
        grid_rtol: Relative tolerance for strict grid checks.
        grid_atol: Absolute tolerance for strict grid checks.
        dtype: NumPy dtype string for output arrays.

    Returns:
        Path to merged HDF5 output.

    Raises:
        ValueError: If argument lengths are inconsistent.
    """
    # validate required inputs
    if not input_files:
        raise ValueError("merge_files_to_hdf5 requires at least one input file")

    n_files = len(input_files)

    # build per-file grid list
    if input_grids is None:
        grids = [None] * n_files
    elif len(input_grids) == 1 and n_files > 1:
        grids = [input_grids[0]] * n_files
    elif len(input_grids) == n_files:
        grids = list(input_grids)
    else:
        raise ValueError("input_grids must have length 1 or match number of input_files")

    # build per-file iteration list
    if input_iterations is None:
        iterations = [1] * n_files
    elif len(input_iterations) == 1 and n_files > 1:
        iterations = [int(input_iterations[0])] * n_files
    elif len(input_iterations) == n_files:
        iterations = [int(v) for v in input_iterations]
    else:
        raise ValueError("input_iterations must have length 1 or match number of input_files")

    # read each source dataset
    datasets: list[Dataset] = []
    for fpath, gpath, it in zip(input_files, grids, iterations):
        dataset = read_file(fpath=fpath, grid_file=gpath, it=it)
        datasets.append(dataset)

    # merge snapshots into timestep-indexed flow blocks
    base_dataset, timestep_flow = merge_datasets(
        datasets,
        timestep_ids=iterations,
        strict_grid=strict_grid,
        grid_rtol=grid_rtol,
        grid_atol=grid_atol,
    )

    # track source-file metadata in output attrs
    attrs_out = copy_attrs(base_dataset.attrs)
    attrs_out["cfd_ops_merge_sources"] = "|".join(str(Path(p)) for p in input_files)
    base_dataset.attrs = attrs_out

    # write output grouped HDF5 file
    return write_merged_hdf5(
        output_path=output_file,
        base_dataset=base_dataset,
        timestep_flow=timestep_flow,
        dtype=dtype,
    )
