"""Command-line interface for cfd-ops."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from cfd_io import read_file, write_file

from cfd_ops.operations import (
    cut_dataset,
    extend_dataset,
    merge_files_to_hdf5,
    rotate_dataset,
    translate_dataset,
    transpose_dataset,
)

# --------------------------------------------------
# app and helpers
# --------------------------------------------------
app = typer.Typer(
    name="cfd-ops",
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
    help="Apply cfd-ops transformations to cfd-io datasets.",
)


@app.callback()
def _app_callback(ctx: typer.Context) -> None:
    """Show help when no subcommand is provided."""
    # show top-level help and exit cleanly when no command is selected
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


def _read_dataset(
    input_path: Path,
    *,
    input_grid: Path | None,
    it: int,
):
    """Read one dataset with optional split-format options."""
    # read dataset with optional grid companion and timestep index
    return read_file(input_path, grid_file=input_grid, it=it)


def _write_dataset(
    output_path: Path,
    dataset,
    *,
    output_grid: Path | None,
) -> None:
    """Write one dataset with optional split-format options."""
    # write dataset with optional grid companion
    write_file(output_path, dataset, grid_file=output_grid)


# --------------------------------------------------
# commands
# --------------------------------------------------
@app.command("rotate")
def rotate_command(
    input_path: Annotated[Path, typer.Argument(help="Input dataset file")],
    output_path: Annotated[Path, typer.Argument(help="Output dataset file")],
    axis: Annotated[str, typer.Option("--axis", help="Rotation axis: x, y, or z")],
    angle: Annotated[float, typer.Option("--angle", help="Rotation angle in degrees")],
    origin: Annotated[
        tuple[float, float, float],
        typer.Option(
            "--origin",
            help="Rotation origin coordinates as OX OY OZ",
        ),
    ] = (0.0, 0.0, 0.0),
    no_rotate_flow: Annotated[
        bool,
        typer.Option("--no-rotate-flow", help="Do not rotate recognized velocity components"),
    ] = False,
    input_grid: Annotated[
        Path | None,
        typer.Option("--grid-in", "-g", help="Companion grid file for split-format input"),
    ] = None,
    output_grid: Annotated[
        Path | None,
        typer.Option("--grid-out", "-G", help="Companion grid file for split-format output"),
    ] = None,
    it: Annotated[int, typer.Option("--it", help="Input timestep index for split-format input")] = 1,
) -> None:
    """Rotate dataset coordinates and vector fields.

    Recognized velocity components (uvel/vvel/wvel and similar pairs) are
    rotated together with the grid coordinates unless --no-rotate-flow
    is given. Format conversion happens for free when input and output
    extensions differ.

    \b
    Examples:
        # rotate 30 deg about z, also rotate velocity vectors
        cfd-ops rotate in.h5 out.h5 --axis z --angle 30

        # rotate about a non-origin pivot
        cfd-ops rotate in.h5 out.h5 --axis y --angle 15 \\
            --origin 0.5 0.0 0.0

        # rotate grid only, leave flow components untouched
        cfd-ops rotate in.h5 out.h5 --axis x --angle 90 --no-rotate-flow

        # split-format in, HDF5 out
        cfd-ops rotate flow.s8 out.h5 --axis z --angle 45 -g grid.s8
    """
    # read input dataset
    dataset = _read_dataset(input_path, input_grid=input_grid, it=it)

    # rotate dataset
    rotated = rotate_dataset(
        dataset,
        axis=axis,
        angle_deg=angle,
        origin=(origin[0], origin[1], origin[2]),
        rotate_flow=not no_rotate_flow,
    )

    # write transformed dataset
    _write_dataset(output_path, rotated, output_grid=output_grid)


@app.command("cut")
def cut_command(
    input_path: Annotated[Path, typer.Argument(help="Input dataset file")],
    output_path: Annotated[Path, typer.Argument(help="Output dataset file")],
    i_s: Annotated[
        int | None,
        typer.Option("--is", help="Streamwise (i) start, 1-based inclusive (default: 1)"),
    ] = None,
    i_e: Annotated[
        int | None,
        typer.Option("--ie", help="Streamwise (i) end, 1-based inclusive (default: ni)"),
    ] = None,
    di: Annotated[
        int,
        typer.Option("--di", help="Streamwise (i) stride (default: 1)"),
    ] = 1,
    j_s: Annotated[
        int | None,
        typer.Option("--js", help="Wall-normal (j) start, 1-based inclusive (default: 1)"),
    ] = None,
    j_e: Annotated[
        int | None,
        typer.Option("--je", help="Wall-normal (j) end, 1-based inclusive (default: nj)"),
    ] = None,
    dj: Annotated[
        int,
        typer.Option("--dj", help="Wall-normal (j) stride (default: 1)"),
    ] = 1,
    k_s: Annotated[
        int | None,
        typer.Option("--ks", help="Spanwise (k) start, 1-based inclusive (default: 1)"),
    ] = None,
    k_e: Annotated[
        int | None,
        typer.Option("--ke", help="Spanwise (k) end, 1-based inclusive (default: nk)"),
    ] = None,
    dk: Annotated[
        int,
        typer.Option("--dk", help="Spanwise (k) stride (default: 1)"),
    ] = 1,
    input_grid: Annotated[
        Path | None,
        typer.Option("--grid-in", "-g", help="Companion grid file for split-format input"),
    ] = None,
    output_grid: Annotated[
        Path | None,
        typer.Option("--grid-out", "-G", help="Companion grid file for split-format output"),
    ] = None,
    it: Annotated[int, typer.Option("--it", help="Input timestep index for split-format input")] = 1,
) -> None:
    """Cut/subset dataset along i/j/k.

    Indices are 1-based inclusive on both ends, matching Fortran/Tecplot
    conventions. Omitted bounds keep the full extent on that axis. If
    no flags are given on an axis, that axis is left untouched.

    Examples:
        # trim the first 50 streamwise stations
        cfd-ops cut in.h5 out.h5 --is 51

        # keep streamwise indices 100..500 with stride 2
        cfd-ops cut in.h5 out.h5 --is 100 --ie 500 --di 2

        # extract a single wall-normal index (j=10)
        cfd-ops cut in.h5 out.h5 --js 10 --je 10
    """
    # validate strides
    if di < 1 or dj < 1 or dk < 1:
        raise typer.BadParameter("strides --di/--dj/--dk must be >= 1")

    # read input dataset
    dataset = _read_dataset(input_path, input_grid=input_grid, it=it)

    # apply cuts axis-by-axis -- skip axes with no flags
    # in-memory axis convention is (ni, nj, nk) so axis name maps directly
    cut_data = dataset
    for axis_name, start_1b, end_1b, stride in (
        ("i", i_s, i_e, di),
        ("j", j_s, j_e, dj),
        ("k", k_s, k_e, dk),
    ):
        # skip axis when no slicing requested
        if start_1b is None and end_1b is None and stride == 1:
            continue

        # convert 1-based inclusive bounds to 0-based half-open python slice
        # start: 1-based inclusive  -> 0-based: start_1b - 1
        # end:   1-based inclusive  -> 0-based half-open: end_1b
        py_start = (start_1b - 1) if start_1b is not None else None
        py_stop = end_1b if end_1b is not None else None

        cut_data = cut_dataset(
            cut_data,
            axis=axis_name,
            start=py_start,
            stop=py_stop,
            step=stride,
        )

    # write transformed dataset
    _write_dataset(output_path, cut_data, output_grid=output_grid)


@app.command("translate")
def translate_command(
    input_path: Annotated[Path, typer.Argument(help="Input dataset file")],
    output_path: Annotated[Path, typer.Argument(help="Output dataset file")],
    dx: Annotated[float, typer.Option("--dx", help="Translation in x")] = 0.0,
    dy: Annotated[float, typer.Option("--dy", help="Translation in y")] = 0.0,
    dz: Annotated[float, typer.Option("--dz", help="Translation in z")] = 0.0,
    input_grid: Annotated[
        Path | None,
        typer.Option("--grid-in", "-g", help="Companion grid file for split-format input"),
    ] = None,
    output_grid: Annotated[
        Path | None,
        typer.Option("--grid-out", "-G", help="Companion grid file for split-format output"),
    ] = None,
    it: Annotated[int, typer.Option("--it", help="Input timestep index for split-format input")] = 1,
) -> None:
    """Translate dataset coordinates by constant offsets.

    Flow variables are unchanged; only grid coordinates shift. Format
    conversion happens for free when input and output extensions differ.
    Grid flags are required only for split-format files (.s8/.s4/.cd).

    \b
    Examples:
        # shift downstream by 1.0 in x (HDF5 round-trip)
        cfd-ops translate in.h5 out.h5 --dx 1.0

        # split-format in, HDF5 out
        cfd-ops translate flow.s8 out.h5 --dx 0.5 -g grid.s8

        # HDF5 in, split-format out
        cfd-ops translate in.h5 flow.s8 --dy -0.1 -G grid.s8

        # multi-axis shift
        cfd-ops translate in.h5 out.h5 --dx 1.0 --dy 0.5 --dz -0.25
    """
    # read input dataset
    dataset = _read_dataset(input_path, input_grid=input_grid, it=it)

    # translate dataset
    translated = translate_dataset(
        dataset,
        dx=dx,
        dy=dy,
        dz=dz,
    )

    # write transformed dataset
    _write_dataset(output_path, translated, output_grid=output_grid)


@app.command("transpose")
def transpose_command(
    input_path: Annotated[Path, typer.Argument(help="Input dataset file")],
    output_path: Annotated[Path, typer.Argument(help="Output dataset file")],
    axes: Annotated[
        str,
        typer.Option(
            "--axes",
            help="Axis pair to swap: ij, ik, or jk (order ignored)",
        ),
    ],
    input_grid: Annotated[
        Path | None,
        typer.Option("--grid-in", "-g", help="Companion grid file for split-format input"),
    ] = None,
    output_grid: Annotated[
        Path | None,
        typer.Option("--grid-out", "-G", help="Companion grid file for split-format output"),
    ] = None,
    it: Annotated[int, typer.Option("--it", help="Input timestep index for split-format input")] = 1,
) -> None:
    """Transpose two structured grid axes (i, j, k).

    Swaps the requested pair of axes in grid coordinates and in every
    flow field with at least three dimensions. Useful when an external
    file uses a transposed orientation relative to your downstream
    pipeline (e.g. rows/columns reversed in 2-D data).

    \b
    Examples:
        # swap streamwise and wall-normal axes
        cfd-ops transpose in.h5 out.h5 --axes ij

        # swap streamwise and spanwise axes
        cfd-ops transpose in.h5 out.h5 --axes ik

        # swap wall-normal and spanwise axes
        cfd-ops transpose in.h5 out.h5 --axes jk
    """
    # read input dataset
    dataset = _read_dataset(input_path, input_grid=input_grid, it=it)

    # transpose requested axis pair
    transposed = transpose_dataset(dataset, axes=axes)

    # write transformed dataset
    _write_dataset(output_path, transposed, output_grid=output_grid)


@app.command("extend")
def extend_command(
    input_path: Annotated[Path, typer.Argument(help="Input dataset file")],
    output_path: Annotated[Path, typer.Argument(help="Output dataset file")],
    axis: Annotated[str, typer.Option("--axis", help="Extension axis: i, j, or k")],
    before: Annotated[int, typer.Option("--before", help="Cells to prepend")] = 0,
    after: Annotated[int, typer.Option("--after", help="Cells to append")] = 0,
    mode: Annotated[str, typer.Option("--mode", help="Padding mode: edge or constant")] = "edge",
    constant_value: Annotated[
        float,
        typer.Option("--constant-value", help="Fill value used for constant mode"),
    ] = 0.0,
    input_grid: Annotated[
        Path | None,
        typer.Option("--grid-in", "-g", help="Companion grid file for split-format input"),
    ] = None,
    output_grid: Annotated[
        Path | None,
        typer.Option("--grid-out", "-G", help="Companion grid file for split-format output"),
    ] = None,
    it: Annotated[int, typer.Option("--it", help="Input timestep index for split-format input")] = 1,
) -> None:
    """Extend dataset by padding along one index axis.

    Pads ``before`` cells before index 0 and ``after`` cells after the
    last index along the requested axis. Mode ``edge`` repeats the
    boundary slice; mode ``constant`` fills with --constant-value.

    \b
    Examples:
        # add 5 ghost cells before i=0 (edge replication)
        cfd-ops extend in.h5 out.h5 --axis i --before 5

        # pad both ends in j with constant zeros
        cfd-ops extend in.h5 out.h5 --axis j --before 2 --after 2 \\
            --mode constant --constant-value 0.0

        # add 10 cells after the last spanwise index
        cfd-ops extend in.h5 out.h5 --axis k --after 10
    """
    # read input dataset
    dataset = _read_dataset(input_path, input_grid=input_grid, it=it)

    # extend dataset
    extended = extend_dataset(
        dataset,
        axis=axis,
        before=before,
        after=after,
        mode=mode,
        constant_value=constant_value,
    )

    # write transformed dataset
    _write_dataset(output_path, extended, output_grid=output_grid)


@app.command("merge")
def merge_command(
    output_path: Annotated[Path, typer.Argument(help="Merged output HDF5 file")],
    input_files: Annotated[
        list[Path],
        typer.Option("--input", help="Input files in merge order; repeat flag for multiple files"),
    ],
    grids: Annotated[
        list[Path] | None,
        typer.Option("--grid", help="Companion split-format grid files; repeat as needed"),
    ] = None,
    iterations: Annotated[
        list[int] | None,
        typer.Option("--it", help="Input timestep indices; repeat as needed"),
    ] = None,
    allow_grid_drift: Annotated[
        bool,
        typer.Option("--allow-grid-drift", help="Only enforce matching grid shapes"),
    ] = False,
    grid_rtol: Annotated[float, typer.Option("--grid-rtol", help="Relative tolerance for grid checks")] = 1.0e-10,
    grid_atol: Annotated[float, typer.Option("--grid-atol", help="Absolute tolerance for grid checks")] = 1.0e-12,
    dtype: Annotated[str, typer.Option("--dtype", help="Output dtype for HDF5 datasets")] = "f",
) -> None:
    """Merge snapshots from multiple files into one multi-timestep HDF5 file.

    Each --input file contributes one timestep. Grids are checked for
    consistency (drift tolerated under --allow-grid-drift).

    \b
    Examples:
        # merge three HDF5 snapshots
        cfd-ops merge merged.h5 \\
            --input snap_001.h5 --input snap_002.h5 --input snap_003.h5

        # merge with companion grids for split-format inputs
        cfd-ops merge merged.h5 \\
            --input flow_001.s8 --grid grid.s8 \\
            --input flow_002.s8 --grid grid.s8

        # tolerate small grid drift between snapshots
        cfd-ops merge merged.h5 \\
            --input a.h5 --input b.h5 \\
            --allow-grid-drift --grid-rtol 1e-8
    """
    # validate required input list
    if len(input_files) == 0:
        raise typer.BadParameter("--input requires at least one file")

    # merge multiple snapshots into one multi-timestep output
    merge_files_to_hdf5(
        input_files=list(input_files),
        output_file=output_path,
        input_grids=None if grids in (None, []) else list(grids),
        input_iterations=None if iterations in (None, []) else list(iterations),
        strict_grid=not allow_grid_drift,
        grid_rtol=grid_rtol,
        grid_atol=grid_atol,
        dtype=dtype,
    )

# --------------------------------------------------
# public API
# --------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Run Typer CLI and return process exit code."""
    # run typer app with standalone mode so click/typer handle help paths cleanly
    try:
        app(args=argv, prog_name="cfd-ops", standalone_mode=True)
        return 0
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
