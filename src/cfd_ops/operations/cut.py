"""Cut/subset operations for structured CFD datasets."""

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
IndexAxis = Literal["i", "j", "k"]


# --------------------------------------------------
# public API
# --------------------------------------------------
def cut_dataset(
    dataset: Dataset,
    *,
    axis: IndexAxis,
    index: int | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> Dataset:
    """Cut or subset a structured dataset along i/j/k.

    Args:
        dataset: Input cfd-io dataset.
        axis: Index axis (i, j, or k).
        index: Single index to extract. Keeps dimensionality.
        start: Slice start for range mode.
        stop: Slice stop for range mode.
        step: Slice step for range mode.

    Returns:
        Cut dataset.

    Raises:
        TypeError: If grid is not structured.
        ValueError: If cut parameters are invalid.
    """
    # validate inputs
    if axis not in ("i", "j", "k"):
        raise ValueError(f"invalid cut axis '{axis}'; expected one of i,j,k")

    # validate grid type
    grid = require_structured_grid(dataset, "cut_dataset")

    # validate mutually exclusive parameter styles
    if index is not None and any(v is not None for v in (start, stop, step)):
        raise ValueError("use either index or start/stop/step, not both")

    if index is None and start is None and stop is None and step is None:
        raise ValueError("cut requires either index or start/stop/step")

    # map axis label to ndarray axis index
    axis_map = {"i": 0, "j": 1, "k": 2}
    ax = axis_map[axis]

    # build slicing object
    slicer = [slice(None), slice(None), slice(None)]
    if index is not None:
        slicer[ax] = slice(index, index + 1, 1)
    else:
        slicer[ax] = slice(start, stop, step)
    cut = tuple(slicer)

    # cut grid coordinates
    grid_out = StructuredGrid(
        x=np.array(grid.x[cut], copy=True),
        y=np.array(grid.y[cut], copy=True),
        z=np.array(grid.z[cut], copy=True),
    )

    # cut flow variables where dimensionality is compatible
    flow_out: dict[str, Field] = {}
    for name, value in dataset.flow.items():
        field_obj = as_field(value)
        data = np.asarray(field_obj.data)

        # cut arrays that are at least 3-D and align with grid axes
        if data.ndim >= 3:
            data_cut = np.array(data[cut], copy=True)
        else:
            data_cut = np.array(data, copy=True)

        flow_out[name] = Field(data=data_cut, association=field_obj.association)

    # copy and annotate attrs
    attrs_out = copy_attrs(dataset.attrs)
    attrs_out["cfd_ops_cut_axis"] = axis
    if index is not None:
        attrs_out["cfd_ops_cut_index"] = int(index)
    else:
        # store slice as a string to keep hdf5 attribute types simple
        # (h5py can't write tuples of mixed types or python None)
        def _fmt(v: int | None) -> str:
            return "None" if v is None else str(int(v))
        attrs_out["cfd_ops_cut_slice"] = (
            f"{_fmt(start)}:{_fmt(stop)}:{_fmt(step)}"
        )

    return Dataset(grid=grid_out, flow=flow_out, attrs=attrs_out)
