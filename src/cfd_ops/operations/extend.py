"""Domain extension operation for structured CFD datasets."""

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
PadMode = Literal["edge", "constant"]


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _build_pad_width(axis_index: int, before: int, after: int, ndim: int) -> list[tuple[int, int]]:
    """Build pad-width specification for NumPy arrays."""
    # start with no padding in all dimensions
    pad = [(0, 0) for _ in range(ndim)]

    # apply padding only on selected axis
    pad[axis_index] = (before, after)

    return pad


def _pad_array(
    data: np.ndarray,
    *,
    axis_index: int,
    before: int,
    after: int,
    mode: PadMode,
    constant_value: float,
) -> np.ndarray:
    """Pad one array along selected axis with configured mode."""
    # keep original when no extension requested
    if before == 0 and after == 0:
        return np.array(data, copy=True)

    # build padding configuration
    pad_width = _build_pad_width(axis_index=axis_index, before=before, after=after, ndim=data.ndim)

    # pad according to selected strategy
    if mode == "edge":
        return np.pad(data, pad_width=pad_width, mode="edge")

    return np.pad(
        data,
        pad_width=pad_width,
        mode="constant",
        constant_values=constant_value,
    )


# --------------------------------------------------
# public API
# --------------------------------------------------
def extend_dataset(
    dataset: Dataset,
    *,
    axis: IndexAxis,
    before: int = 0,
    after: int = 0,
    mode: PadMode = "edge",
    constant_value: float = 0.0,
) -> Dataset:
    """Extend a structured dataset along i/j/k by padding.

    Args:
        dataset: Input cfd-io dataset.
        axis: Extension axis (i, j, or k).
        before: Number of cells to prepend.
        after: Number of cells to append.
        mode: Padding mode. "edge" repeats boundary values; "constant"
            fills with constant_value.
        constant_value: Fill value used when mode is "constant".

    Returns:
        Extended dataset.

    Raises:
        TypeError: If grid is not structured.
        ValueError: If parameters are invalid.
    """
    # validate axis and mode
    if axis not in ("i", "j", "k"):
        raise ValueError(f"invalid extend axis '{axis}'; expected one of i,j,k")

    if mode not in ("edge", "constant"):
        raise ValueError(f"invalid extend mode '{mode}'; expected edge or constant")

    # validate extension counts
    if before < 0 or after < 0:
        raise ValueError("extend before/after must be >= 0")

    if before == 0 and after == 0:
        raise ValueError("extend requires before > 0 or after > 0")

    # validate grid type
    grid = require_structured_grid(dataset, "extend_dataset")

    # map axis label to ndarray axis index
    axis_map = {"i": 0, "j": 1, "k": 2}
    axis_index = axis_map[axis]

    # extend grid coordinates
    grid_out = StructuredGrid(
        x=_pad_array(
            grid.x,
            axis_index=axis_index,
            before=before,
            after=after,
            mode=mode,
            constant_value=constant_value,
        ),
        y=_pad_array(
            grid.y,
            axis_index=axis_index,
            before=before,
            after=after,
            mode=mode,
            constant_value=constant_value,
        ),
        z=_pad_array(
            grid.z,
            axis_index=axis_index,
            before=before,
            after=after,
            mode=mode,
            constant_value=constant_value,
        ),
    )

    # extend flow variables where dimensionality is compatible
    flow_out: dict[str, Field] = {}
    for name, value in dataset.flow.items():
        field_obj = as_field(value)
        data = np.asarray(field_obj.data)

        # extend arrays that include grid-index dimensions
        if data.ndim >= 3:
            data_out = _pad_array(
                data,
                axis_index=axis_index,
                before=before,
                after=after,
                mode=mode,
                constant_value=constant_value,
            )
        else:
            data_out = np.array(data, copy=True)

        flow_out[name] = Field(data=data_out, association=field_obj.association)

    # copy and annotate attrs
    attrs_out = copy_attrs(dataset.attrs)
    attrs_out["cfd_ops_extend_axis"] = axis
    attrs_out["cfd_ops_extend_before"] = int(before)
    attrs_out["cfd_ops_extend_after"] = int(after)
    attrs_out["cfd_ops_extend_mode"] = mode

    return Dataset(grid=grid_out, flow=flow_out, attrs=attrs_out)
