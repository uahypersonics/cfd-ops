"""Transpose (axis swap) operation for structured CFD datasets."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from typing import Literal

import numpy as np
from cfd_io import Dataset, Field, StructuredGrid

from cfd_ops.operations.common import as_field, copy_attrs, require_structured_grid

# --------------------------------------------------
# type aliases
# --------------------------------------------------
AxisPair = Literal["ij", "ji", "ik", "ki", "jk", "kj"]


# --------------------------------------------------
# public API
# --------------------------------------------------
def transpose_dataset(
    dataset: Dataset,
    *,
    axes: AxisPair,
) -> Dataset:
    """Transpose two of the three structured grid axes (i, j, k).

    Swaps the requested pair of axes in the grid coordinates and in
    every flow field with at least three dimensions. Lower-rank flow
    arrays (e.g. 1-D wall data) pass through unchanged.

    Args:
        dataset: Input cfd-io dataset.
        axes: Axis pair to swap. Order does not matter -- ``"ij"`` and
            ``"ji"`` both swap i and j. Valid values: ``ij``, ``ik``, ``jk``
            (and their reverses).

    Returns:
        New dataset with the requested axes swapped.

    Raises:
        TypeError: If the grid is not structured.
        ValueError: If ``axes`` is not a recognized pair.
    """
    # map axis-pair label -> two ndarray axis indices to swap
    pair_map: dict[str, tuple[int, int]] = {
        "ij": (0, 1), "ji": (0, 1),
        "ik": (0, 2), "ki": (0, 2),
        "jk": (1, 2), "kj": (1, 2),
    }

    # validate inputs
    if axes not in pair_map:
        raise ValueError(
            f"invalid transpose axes '{axes}'; expected one of "
            "ij, ji, ik, ki, jk, kj"
        )
    a0, a1 = pair_map[axes]

    # validate grid type
    grid = require_structured_grid(dataset, "transpose_dataset")

    # swap axes in grid coordinates
    grid_out = StructuredGrid(
        x=np.ascontiguousarray(np.swapaxes(grid.x, a0, a1)),
        y=np.ascontiguousarray(np.swapaxes(grid.y, a0, a1)),
        z=np.ascontiguousarray(np.swapaxes(grid.z, a0, a1)),
    )

    # swap axes in every flow field with at least 3 dimensions
    flow_out: dict[str, Field] = {}
    for name, value in dataset.flow.items():
        field_obj = as_field(value)
        data = np.asarray(field_obj.data)

        # transpose only structured (3-D+) arrays; pass-through otherwise
        if data.ndim >= 3:
            data_out = np.ascontiguousarray(np.swapaxes(data, a0, a1))
        else:
            data_out = np.array(data, copy=True)

        flow_out[name] = Field(data=data_out, association=field_obj.association)

    # copy attrs and annotate operation details
    attrs_out = copy_attrs(dataset.attrs)
    # canonical lower-axis-first label for traceability
    canonical = "".join(sorted(axes))
    attrs_out["cfd_ops_transpose_axes"] = canonical

    return Dataset(grid=grid_out, flow=flow_out, attrs=attrs_out)
