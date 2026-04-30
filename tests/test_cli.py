"""Tests for cfd-op command-line interface."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
import numpy as np
from cfd_io import Dataset, Field, StructuredGrid

from cfd_ops import cli


# --------------------------------------------------
# fixtures/helpers
# --------------------------------------------------
def _sample_dataset() -> Dataset:
    """Create a tiny dataset used by CLI tests."""
    # build tiny grid
    x = np.zeros((2, 2, 2), dtype=float)
    y = np.zeros((2, 2, 2), dtype=float)
    z = np.zeros((2, 2, 2), dtype=float)
    grid = StructuredGrid(x=x, y=y, z=z)

    # build tiny flow
    flow = {
        "u": Field(data=np.ones((2, 2, 2), dtype=float), association="node"),
        "v": Field(data=np.zeros((2, 2, 2), dtype=float), association="node"),
        "w": Field(data=np.zeros((2, 2, 2), dtype=float), association="node"),
    }

    return Dataset(grid=grid, flow=flow, attrs={})


# --------------------------------------------------
# rotate command
# --------------------------------------------------
def test_cli_rotate_dispatch(monkeypatch) -> None:
    """CLI rotate command should read, transform, and write once."""
    # build recording state
    called: dict[str, object] = {}
    dataset = _sample_dataset()

    # patch read/write/rotate functions
    def fake_read(path, grid_file=None, it=1):
        called["read"] = {"path": path, "grid_file": grid_file, "it": it}
        return dataset

    def fake_rotate(ds, axis, angle_deg, origin, rotate_flow):
        called["rotate"] = {
            "axis": axis,
            "angle_deg": angle_deg,
            "origin": origin,
            "rotate_flow": rotate_flow,
            "same_dataset": ds is dataset,
        }
        return dataset

    def fake_write(path, ds, grid_file=None):
        called["write"] = {
            "path": path,
            "grid_file": grid_file,
            "same_dataset": ds is dataset,
        }
        return path

    monkeypatch.setattr(cli, "read_file", fake_read)
    monkeypatch.setattr(cli, "rotate_dataset", fake_rotate)
    monkeypatch.setattr(cli, "write_file", fake_write)

    # run CLI command
    code = cli.main(
        [
            "rotate",
            "in.cgns",
            "out.cgns",
            "--axis",
            "z",
            "--angle",
            "90",
            "--origin",
            "1",
            "2",
            "3",
        ]
    )

    # check exit status and dispatch details
    assert code == 0
    assert called["rotate"]["axis"] == "z"
    assert called["rotate"]["angle_deg"] == 90.0
    assert called["rotate"]["origin"] == (1.0, 2.0, 3.0)
    assert called["rotate"]["rotate_flow"] is True
    assert called["rotate"]["same_dataset"] is True
    assert called["write"]["same_dataset"] is True


# --------------------------------------------------
# cut command
# --------------------------------------------------
def test_cli_cut_single_index_dispatch(monkeypatch) -> None:
    """CLI cut with --js N --je N should call cut_dataset for axis j only."""
    # build recording state
    cut_calls: list[dict[str, object]] = []
    dataset = _sample_dataset()

    # patch read/write/cut functions
    def fake_read(path, grid_file=None, it=1):
        return dataset

    def fake_cut(ds, axis, index=None, start=None, stop=None, step=None):
        cut_calls.append({
            "axis": axis,
            "start": start,
            "stop": stop,
            "step": step,
        })
        return ds

    def fake_write(path, ds, grid_file=None):
        return path

    monkeypatch.setattr(cli, "read_file", fake_read)
    monkeypatch.setattr(cli, "cut_dataset", fake_cut)
    monkeypatch.setattr(cli, "write_file", fake_write)

    # run CLI: extract a single wall-normal index j=3 (1-based inclusive both ends)
    code = cli.main(["cut", "in.cgns", "out.cgns", "--js", "3", "--je", "3"])

    # check that exactly one cut on axis j was issued with python-style 0-based half-open bounds
    assert code == 0
    assert len(cut_calls) == 1
    assert cut_calls[0]["axis"] == "j"
    assert cut_calls[0]["start"] == 2  # 1-based 3 -> 0-based 2
    assert cut_calls[0]["stop"] == 3   # inclusive end 3 -> half-open stop 3
    assert cut_calls[0]["step"] == 1


def test_cli_cut_open_bounds_streamwise(monkeypatch) -> None:
    """CLI cut with only --is N should pass start=N-1, stop=None on axis i."""
    # build recording state
    cut_calls: list[dict[str, object]] = []
    dataset = _sample_dataset()

    # patch read/write/cut functions
    def fake_read(path, grid_file=None, it=1):
        return dataset

    def fake_cut(ds, axis, index=None, start=None, stop=None, step=None):
        cut_calls.append({
            "axis": axis,
            "start": start,
            "stop": stop,
            "step": step,
        })
        return ds

    def fake_write(path, ds, grid_file=None):
        return path

    monkeypatch.setattr(cli, "read_file", fake_read)
    monkeypatch.setattr(cli, "cut_dataset", fake_cut)
    monkeypatch.setattr(cli, "write_file", fake_write)

    # trim first 50 streamwise stations: keep i=51..ni
    code = cli.main(["cut", "in.cgns", "out.cgns", "--is", "51"])

    assert code == 0
    assert len(cut_calls) == 1
    assert cut_calls[0]["axis"] == "i"
    assert cut_calls[0]["start"] == 50  # 1-based 51 -> 0-based 50
    assert cut_calls[0]["stop"] is None
    assert cut_calls[0]["step"] == 1


def test_cli_cut_multi_axis_dispatch(monkeypatch) -> None:
    """CLI cut with i and j flags should issue two sequential cut_dataset calls."""
    # build recording state
    cut_calls: list[dict[str, object]] = []
    dataset = _sample_dataset()

    # patch read/write/cut functions
    def fake_read(path, grid_file=None, it=1):
        return dataset

    def fake_cut(ds, axis, index=None, start=None, stop=None, step=None):
        cut_calls.append({
            "axis": axis,
            "start": start,
            "stop": stop,
            "step": step,
        })
        return ds

    def fake_write(path, ds, grid_file=None):
        return path

    monkeypatch.setattr(cli, "read_file", fake_read)
    monkeypatch.setattr(cli, "cut_dataset", fake_cut)
    monkeypatch.setattr(cli, "write_file", fake_write)

    # cut both i and j at once
    code = cli.main([
        "cut", "in.cgns", "out.cgns",
        "--is", "10", "--ie", "100", "--di", "2",
        "--js", "5", "--je", "20",
    ])

    assert code == 0
    assert len(cut_calls) == 2
    assert cut_calls[0]["axis"] == "i"
    assert cut_calls[0]["start"] == 9
    assert cut_calls[0]["stop"] == 100
    assert cut_calls[0]["step"] == 2
    assert cut_calls[1]["axis"] == "j"
    assert cut_calls[1]["start"] == 4
    assert cut_calls[1]["stop"] == 20
    assert cut_calls[1]["step"] == 1


def test_cli_cut_no_flags_is_noop(monkeypatch) -> None:
    """CLI cut with no axis flags should not call cut_dataset at all."""
    # build recording state
    cut_calls: list[dict[str, object]] = []
    dataset = _sample_dataset()

    # patch read/write/cut functions
    def fake_read(path, grid_file=None, it=1):
        return dataset

    def fake_cut(ds, axis, index=None, start=None, stop=None, step=None):
        cut_calls.append({"axis": axis})
        return ds

    def fake_write(path, ds, grid_file=None):
        return path

    monkeypatch.setattr(cli, "read_file", fake_read)
    monkeypatch.setattr(cli, "cut_dataset", fake_cut)
    monkeypatch.setattr(cli, "write_file", fake_write)

    # run with no axis flags -- should be a pass-through
    code = cli.main(["cut", "in.cgns", "out.cgns"])

    assert code == 0
    assert cut_calls == []


def test_cli_translate_dispatch(monkeypatch) -> None:
    """CLI translate command should dispatch with dx/dy/dz offsets."""
    # build recording state
    called: dict[str, object] = {}
    dataset = _sample_dataset()

    # patch read/write/translate functions
    def fake_read(path, grid_file=None, it=1):
        called["read"] = {"path": path, "grid_file": grid_file, "it": it}
        return dataset

    def fake_translate(ds, dx, dy, dz):
        called["translate"] = {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "same_dataset": ds is dataset,
        }
        return dataset

    def fake_write(path, ds, grid_file=None):
        called["write"] = {
            "path": path,
            "grid_file": grid_file,
            "same_dataset": ds is dataset,
        }
        return path

    monkeypatch.setattr(cli, "read_file", fake_read)
    monkeypatch.setattr(cli, "translate_dataset", fake_translate)
    monkeypatch.setattr(cli, "write_file", fake_write)

    # run CLI command
    code = cli.main(
        [
            "translate",
            "in.cgns",
            "out.cgns",
            "--dx",
            "1.0",
            "--dy",
            "-2.5",
            "--dz",
            "0.25",
        ]
    )

    # check exit status and dispatch details
    assert code == 0
    assert called["translate"]["dx"] == 1.0
    assert called["translate"]["dy"] == -2.5
    assert called["translate"]["dz"] == 0.25
    assert called["translate"]["same_dataset"] is True
    assert called["write"]["same_dataset"] is True


def test_cli_extend_dispatch(monkeypatch) -> None:
    """CLI extend command should dispatch with padding arguments."""
    # build recording state
    called: dict[str, object] = {}
    dataset = _sample_dataset()

    # patch read/write/extend functions
    def fake_read(path, grid_file=None, it=1):
        called["read"] = {"path": path, "grid_file": grid_file, "it": it}
        return dataset

    def fake_extend(ds, axis, before, after, mode, constant_value):
        called["extend"] = {
            "axis": axis,
            "before": before,
            "after": after,
            "mode": mode,
            "constant_value": constant_value,
            "same_dataset": ds is dataset,
        }
        return dataset

    def fake_write(path, ds, grid_file=None):
        called["write"] = {
            "path": path,
            "grid_file": grid_file,
            "same_dataset": ds is dataset,
        }
        return path

    monkeypatch.setattr(cli, "read_file", fake_read)
    monkeypatch.setattr(cli, "extend_dataset", fake_extend)
    monkeypatch.setattr(cli, "write_file", fake_write)

    # run CLI command
    code = cli.main(
        [
            "extend",
            "in.cgns",
            "out.cgns",
            "--axis",
            "j",
            "--before",
            "1",
            "--after",
            "2",
            "--mode",
            "constant",
            "--constant-value",
            "4.0",
        ]
    )

    # check exit status and dispatch details
    assert code == 0
    assert called["extend"]["axis"] == "j"
    assert called["extend"]["before"] == 1
    assert called["extend"]["after"] == 2
    assert called["extend"]["mode"] == "constant"
    assert called["extend"]["constant_value"] == 4.0
    assert called["extend"]["same_dataset"] is True
    assert called["write"]["same_dataset"] is True


def test_cli_merge_dispatch(monkeypatch) -> None:
    """CLI merge command should forward all merge options correctly."""
    # build recording state
    called: dict[str, object] = {}

    # patch merge function
    def fake_merge(
        *,
        input_files,
        output_file,
        input_grids,
        input_iterations,
        strict_grid,
        grid_rtol,
        grid_atol,
        dtype,
    ):
        called["merge"] = {
            "input_files": input_files,
            "output_file": output_file,
            "input_grids": input_grids,
            "input_iterations": input_iterations,
            "strict_grid": strict_grid,
            "grid_rtol": grid_rtol,
            "grid_atol": grid_atol,
            "dtype": dtype,
        }
        return output_file

    monkeypatch.setattr(cli, "merge_files_to_hdf5", fake_merge)

    # run CLI command
    code = cli.main(
        [
            "merge",
            "merged.h5",
            "--input",
            "f1.s8",
            "--input",
            "f2.s8",
            "--grid",
            "g.s8",
            "--it",
            "1",
            "--it",
            "2",
            "--allow-grid-drift",
            "--grid-rtol",
            "1e-6",
            "--grid-atol",
            "1e-9",
            "--dtype",
            "f8",
        ]
    )

    # check exit status and dispatch details
    assert code == 0
    assert len(called["merge"]["input_files"]) == 2
    assert str(called["merge"]["output_file"]).endswith("merged.h5")
    assert len(called["merge"]["input_grids"]) == 1
    assert called["merge"]["input_iterations"] == [1, 2]
    assert called["merge"]["strict_grid"] is False
    assert called["merge"]["dtype"] == "f8"
