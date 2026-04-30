"""Tests for cfd-ops dataset transformations."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
import numpy as np
from cfd_io import Dataset, Field, StructuredGrid
from h5py import File

from cfd_ops.operations import (
    cut_dataset,
    extend_dataset,
    merge_datasets,
    rotate_dataset,
    translate_dataset,
    write_merged_hdf5,
)


# --------------------------------------------------
# fixtures/helpers
# --------------------------------------------------
def _build_dataset() -> Dataset:
    """Build a simple structured dataset for operation tests."""
    # build structured coordinates
    i, j, k = np.indices((3, 4, 2))
    x = i.astype(float)
    y = j.astype(float)
    z = k.astype(float)
    grid = StructuredGrid(x=x, y=y, z=z)

    # build flow variables
    flow = {
        "u": Field(data=np.ones((3, 4, 2), dtype=float), association="node"),
        "v": Field(data=np.zeros((3, 4, 2), dtype=float), association="node"),
        "w": Field(data=np.zeros((3, 4, 2), dtype=float), association="node"),
        "rho": Field(data=np.full((3, 4, 2), 2.0, dtype=float), association="node"),
    }

    # build dataset
    return Dataset(grid=grid, flow=flow, attrs={"source": "unit-test"})


# --------------------------------------------------
# rotate tests
# --------------------------------------------------
def test_rotate_dataset_z_90_rotates_grid_and_velocity() -> None:
    """Rotate around z by 90 deg and verify coordinates/velocity components."""
    # build input dataset
    dataset = _build_dataset()

    # rotate dataset
    out = rotate_dataset(dataset, axis="z", angle_deg=90.0)

    # check coordinate rotation: x'=-y and y'=x
    assert np.allclose(out.grid.x, -dataset.grid.y)
    assert np.allclose(out.grid.y, dataset.grid.x)
    assert np.allclose(out.grid.z, dataset.grid.z)

    # check velocity vector rotation: [1,0,0] -> [0,1,0]
    assert np.allclose(out.flow["u"].data, 0.0, atol=1e-12)
    assert np.allclose(out.flow["v"].data, 1.0, atol=1e-12)
    assert np.allclose(out.flow["w"].data, 0.0, atol=1e-12)

    # check attrs annotation
    assert out.attrs["cfd_ops_rotate_axis"] == "z"
    assert np.isclose(out.attrs["cfd_ops_rotate_angle_deg"], 90.0)


def test_rotate_dataset_no_rotate_flow_keeps_velocity_components() -> None:
    """Skip flow rotation and verify velocity fields are unchanged."""
    # build input dataset
    dataset = _build_dataset()

    # rotate coordinates only
    out = rotate_dataset(dataset, axis="x", angle_deg=25.0, rotate_flow=False)

    # check coordinate changed
    assert not np.allclose(out.grid.y, dataset.grid.y)

    # check flow unchanged
    assert np.allclose(out.flow["u"].data, dataset.flow["u"].data)
    assert np.allclose(out.flow["v"].data, dataset.flow["v"].data)
    assert np.allclose(out.flow["w"].data, dataset.flow["w"].data)


# --------------------------------------------------
# cut tests
# --------------------------------------------------
def test_cut_dataset_index_keeps_size_one_axis() -> None:
    """Cut by index and verify dimensions and values."""
    # build input dataset
    dataset = _build_dataset()

    # cut at j=2
    out = cut_dataset(dataset, axis="j", index=2)

    # check resulting grid shape
    assert out.grid.x.shape == (3, 1, 2)
    assert out.grid.y.shape == (3, 1, 2)
    assert out.grid.z.shape == (3, 1, 2)

    # check flow cut shape
    assert out.flow["rho"].data.shape == (3, 1, 2)

    # check metadata annotation
    assert out.attrs["cfd_ops_cut_axis"] == "j"
    assert out.attrs["cfd_ops_cut_index"] == 2


def test_cut_dataset_slice_extracts_subrange() -> None:
    """Cut by slice and verify resulting shape and values."""
    # build input dataset
    dataset = _build_dataset()

    # cut i range [1:3]
    out = cut_dataset(dataset, axis="i", start=1, stop=3, step=1)

    # check resulting shape
    assert out.grid.x.shape == (2, 4, 2)
    assert out.flow["u"].data.shape == (2, 4, 2)

    # check value agreement with direct numpy slice
    assert np.allclose(out.grid.x, dataset.grid.x[1:3:1, :, :])

    # check metadata annotation
    assert out.attrs["cfd_ops_cut_axis"] == "i"
    assert out.attrs["cfd_ops_cut_slice"] == "1:3:1"


def test_translate_dataset_shifts_coordinates() -> None:
    """Translate operation should shift grid and preserve flow."""
    # build input dataset
    dataset = _build_dataset()

    # apply translation
    out = translate_dataset(dataset, dx=1.5, dy=-2.0, dz=0.25)

    # check coordinate shifts
    assert np.allclose(out.grid.x, dataset.grid.x + 1.5)
    assert np.allclose(out.grid.y, dataset.grid.y - 2.0)
    assert np.allclose(out.grid.z, dataset.grid.z + 0.25)

    # check flow preserved
    assert np.allclose(out.flow["rho"].data, dataset.flow["rho"].data)


def test_extend_dataset_edge_mode_extends_shape() -> None:
    """Extend operation should pad selected axis and preserve interior."""
    # build input dataset
    dataset = _build_dataset()

    # extend along i-axis with edge padding
    out = extend_dataset(dataset, axis="i", before=1, after=2, mode="edge")

    # check new shape and interior preservation
    assert out.grid.x.shape == (6, 4, 2)
    assert np.allclose(out.grid.x[1:4, :, :], dataset.grid.x)
    assert np.allclose(out.flow["u"].data[1:4, :, :], dataset.flow["u"].data)


def test_merge_datasets_and_write_hdf5(tmp_path) -> None:
    """Merge snapshots and write grouped HDF5 output with two timesteps."""
    # build two snapshots with distinct rho values
    d1 = _build_dataset()
    d2 = _build_dataset()
    d2.flow["rho"].data = np.full((3, 4, 2), 7.0)

    # merge datasets into timestep flow blocks
    base_dataset, timestep_flow = merge_datasets([d1, d2], timestep_ids=[10, 20])

    # write merged output
    out_path = tmp_path / "merged.h5"
    write_merged_hdf5(
        output_path=out_path,
        base_dataset=base_dataset,
        timestep_flow=timestep_flow,
    )

    # verify grouped HDF5 layout and data values
    with File(out_path, "r") as h5obj:
        assert "grid" in h5obj
        assert "flow" in h5obj
        assert "00010" in h5obj["flow"]
        assert "00020" in h5obj["flow"]
        assert np.allclose(h5obj["flow"]["00010"]["rho"][:], 2.0)
        assert np.allclose(h5obj["flow"]["00020"]["rho"][:], 7.0)
