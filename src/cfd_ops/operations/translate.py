"""Translation operation for structured CFD datasets."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import numpy as np
from cfd_io import Dataset, StructuredGrid

from cfd_ops.operations.common import copy_attrs, copy_flow, require_structured_grid


# --------------------------------------------------
# public API
# --------------------------------------------------
def translate_dataset(
    dataset: Dataset,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> Dataset:
    """Translate structured grid coordinates by constant offsets.

    Args:
        dataset: Input cfd-io dataset.
        dx: Translation in x direction.
        dy: Translation in y direction.
        dz: Translation in z direction.

    Returns:
        Translated dataset with unchanged flow variables.

    Raises:
        TypeError: If the grid is not structured.
    """
    # validate grid type
    grid = require_structured_grid(dataset, "translate_dataset")

    # copy flow and attrs so source dataset stays unchanged
    flow_out = copy_flow(dataset.flow)
    attrs_out = copy_attrs(dataset.attrs)

    # apply coordinate shift
    grid_out = StructuredGrid(
        x=np.array(grid.x + float(dx), copy=True),
        y=np.array(grid.y + float(dy), copy=True),
        z=np.array(grid.z + float(dz), copy=True),
    )

    # annotate metadata with operation details
    attrs_out["cfd_ops_translate_dx"] = float(dx)
    attrs_out["cfd_ops_translate_dy"] = float(dy)
    attrs_out["cfd_ops_translate_dz"] = float(dz)

    return Dataset(grid=grid_out, flow=flow_out, attrs=attrs_out)
