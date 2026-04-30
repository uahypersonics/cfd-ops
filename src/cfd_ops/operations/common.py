"""Shared helpers for cfd-ops operations."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
from cfd_io import Dataset, Field, StructuredGrid


# --------------------------------------------------
# flow helpers
# --------------------------------------------------
def as_field(value: Field | np.ndarray, association: str = "node") -> Field:
    """Normalize one flow entry into a Field object.

    Args:
        value: Existing Field or array-like value.
        association: Fallback association if value is not already a Field.

    Returns:
        Field object.
    """
    # convert raw arrays into Field objects
    if isinstance(value, Field):
        return value

    return Field(data=np.asarray(value), association=association)


def copy_flow(flow: dict[str, Field | np.ndarray]) -> dict[str, Field]:
    """Deep-copy flow entries into fresh Field objects.

    Args:
        flow: Mapping of variable name to Field or ndarray.

    Returns:
        New mapping that always contains Field values.
    """
    # build output container
    flow_out: dict[str, Field] = {}

    # copy each flow variable to isolate output from source mutation
    for name, value in flow.items():
        field_obj = as_field(value)
        flow_out[name] = Field(
            data=np.array(field_obj.data, copy=True),
            association=field_obj.association,
        )

    return flow_out


# --------------------------------------------------
# dataset helpers
# --------------------------------------------------
def require_structured_grid(dataset: Dataset, operation: str) -> StructuredGrid:
    """Validate that dataset contains a StructuredGrid.

    Args:
        dataset: Input dataset.
        operation: Operation name for error messages.

    Returns:
        StructuredGrid instance.

    Raises:
        TypeError: If dataset grid is not StructuredGrid.
    """
    # validate supported grid type
    if not isinstance(dataset.grid, StructuredGrid):
        raise TypeError(f"{operation} currently supports StructuredGrid only")

    return dataset.grid


def copy_attrs(attrs: dict[str, Any] | None) -> dict[str, Any]:
    """Create a shallow copy of attrs with safe None handling."""
    # default empty dicts for optional arguments
    return dict(attrs or {})
