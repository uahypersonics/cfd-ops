# cfd-ops

Operations for transforming, extracting, remapping, and combining CFD datasets
built on the [cfd-io](https://github.com/uahypersonics/cfd-io) data model.

[![Test](https://github.com/uahypersonics/cfd-ops/actions/workflows/test.yml/badge.svg)](https://github.com/uahypersonics/cfd-ops/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/uahypersonics/cfd-ops/branch/main/graph/badge.svg)](https://codecov.io/gh/uahypersonics/cfd-ops)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXX) -->
[![PyPI](https://img.shields.io/pypi/v/cfd-ops)](https://pypi.org/project/cfd-ops/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://uahypersonics.github.io/cfd-ops/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Install

```bash
pip install cfd-ops
```

## Documentation

Full documentation: https://uahypersonics.github.io/cfd-ops

## Citation

If `cfd-ops` contributes to published work, please cite via the `CITATION.cff` in this repository.

## Code Style

This project follows established Python community conventions so that
contributors can focus on the physics rather than inventing formatting rules.

| Convention | What it covers | Reference |
|---|---|---|
| [PEP 8](https://peps.python.org/pep-0008/) | Code formatting, naming, whitespace | Python standard style guide |
| [PEP 257](https://peps.python.org/pep-0257/) | Docstring structure (triple-quoted, imperative mood) | Python standard docstring conventions |
| [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) | Docstring sections (`Args`, `Returns`, `Raises`) | Google Python style guide |
| [Ruff](https://docs.astral.sh/ruff/) | Automated linting and formatting | Enforces PEP 8 compliance automatically |
| [typing / TYPE_CHECKING](https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING) | Type hints for IDE support and static analysis | Python standard library |

## Versioning & Releasing

This project uses [Semantic Versioning](https://semver.org/) (`vMAJOR.MINOR.PATCH`):

- **MAJOR** (`v1.0.0`, `v2.0.0`): Breaking API changes
- **MINOR** (`v0.3.0`, `v0.4.0`): New features, backward-compatible
- **PATCH** (`v0.3.1`, `v0.3.2`): Bug fixes, minor corrections

To publish a new version to [PyPI](https://pypi.org/project/cfd-ops/):

1. Commit and push to `main`
2. Tag and push:
   ```bash
   git tag -a vMAJOR.MINOR.PATCH -m "Release vMAJOR.MINOR.PATCH"
   git push origin vMAJOR.MINOR.PATCH
   ```

Use a signed tag when your GPG/SSH signing is configured:

```bash
git tag -s vMAJOR.MINOR.PATCH -m "Release vMAJOR.MINOR.PATCH"
git push origin vMAJOR.MINOR.PATCH
```

The GitHub Actions workflow will automatically build and publish to PyPI via Trusted Publishing.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.
