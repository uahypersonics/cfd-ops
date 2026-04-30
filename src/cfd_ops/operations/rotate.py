"""Rotation operation for structured CFD datasets."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from cfd_io import Dataset, StructuredGrid

from cfd_ops.operations.common import copy_attrs, copy_flow, require_structured_grid

# --------------------------------------------------
# type aliases
# --------------------------------------------------
Axis3D = Literal["x", "y", "z"]


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _rotation_matrix(axis: Axis3D, angle_deg: float) -> np.ndarray:
    """Build a 3x3 right-handed rotation matrix."""
    # compute trig values once
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)

    # select matrix for requested axis
    if axis == "x":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=float,
        )

    if axis == "y":
        return np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=float,
        )

    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _rotate_coordinates(
    grid: StructuredGrid,
    *,
    axis: Axis3D,
    angle_deg: float,
    origin: tuple[float, float, float],
) -> StructuredGrid:
    """Rotate structured-grid coordinates around origin."""
    # build rotation matrix
    matrix = _rotation_matrix(axis=axis, angle_deg=angle_deg)
    ox, oy, oz = origin

    # shift to origin, rotate, then shift back
    xyz = np.stack([grid.x - ox, grid.y - oy, grid.z - oz], axis=0)
    rot = np.einsum("ab,bijk->aijk", matrix, xyz)

    return StructuredGrid(x=rot[0] + ox, y=rot[1] + oy, z=rot[2] + oz)


def _rotate_velocity_components(
    flow: dict[str, object],
    *,
    axis: Axis3D,
    angle_deg: float,
) -> None:
    """Rotate recognized velocity triplets in-place."""
    # build rotation matrix
    matrix = _rotation_matrix(axis=axis, angle_deg=angle_deg)

    # check known velocity naming conventions
    velocity_sets = [
        ("u", "v", "w"),
        ("uvel", "vvel", "wvel"),
        ("vel_x", "vel_y", "vel_z"),
    ]

    # rotate each complete velocity set
    for nx, ny, nz in velocity_sets:
        if nx not in flow or ny not in flow or nz not in flow:
            continue

        vx = flow[nx].data
        vy = flow[ny].data
        vz = flow[nz].data

        vec = np.stack([vx, vy, vz], axis=0)
        vec_rot = np.einsum("ab,b...->a...", matrix, vec)

        flow[nx].data = vec_rot[0]
        flow[ny].data = vec_rot[1]
        flow[nz].data = vec_rot[2]


# --------------------------------------------------
# public API
# --------------------------------------------------
def rotate_dataset(
    dataset: Dataset,
    *,
    axis: Axis3D,
    angle_deg: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotate_flow: bool = True,
) -> Dataset:
    """Rotate a CFD dataset around one Cartesian axis.

    Args:
        dataset: Input cfd-io dataset.
        axis: Rotation axis (x, y, or z).
        angle_deg: Rotation angle in degrees (right-handed).
        origin: Rotation origin coordinates.
        rotate_flow: If True, rotate recognized velocity vectors.

    Returns:
        Rotated dataset.

    Raises:
        TypeError: If the grid is not structured.
        ValueError: If axis is invalid.
    """
    # validate inputs
    if axis not in ("x", "y", "z"):
        raise ValueError(f"invalid axis '{axis}'; expected one of x,y,z")

    # validate grid type
    grid = require_structured_grid(dataset, "rotate_dataset")

    # copy flow and attrs so source dataset stays unchanged
    flow_out = copy_flow(dataset.flow)
    attrs_out = copy_attrs(dataset.attrs)

    # rotate grid coordinates
    grid_out = _rotate_coordinates(
        grid,
        axis=axis,
        angle_deg=angle_deg,
        origin=origin,
    )

    # rotate vector flow fields when requested
    if rotate_flow:
        _rotate_velocity_components(flow_out, axis=axis, angle_deg=angle_deg)

    # annotate metadata with operation details
    attrs_out["cfd_ops_rotate_axis"] = axis
    attrs_out["cfd_ops_rotate_angle_deg"] = float(angle_deg)
    attrs_out["cfd_ops_rotate_origin"] = tuple(float(v) for v in origin)

    return Dataset(grid=grid_out, flow=flow_out, attrs=attrs_out)
