# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python package using a `src` layout. Core package code lives in `src/optimal_checkerboard/`. Algorithm implementations are in `src/optimal_checkerboard/algorithms/`, mesh generation code is in `src/optimal_checkerboard/mesh/`, and shared geometry/data models are in `src/optimal_checkerboard/data_structure/`. Tests are in `tests/` and examples or client entry points are in `script/`. Design notes and algorithm documentation belong in `docs/`.

## Build, Test, and Development Commands

Create a local environment before development:

```bash
python -m venv venv
source venv/bin/activate
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

Install pytest if it is not already available, then run all tests:

```bash
python -m pip install pytest
python -m pytest
```

Run a focused test while iterating:

```bash
python -m pytest tests/test_drag_engine.py
```

Run examples directly, for example:

```bash
python script/example_geometry.py
```

## Coding Style & Naming Conventions

Use Python 3.9+ and 4-space indentation. Follow existing naming: modules and functions use `snake_case`, classes use `PascalCase`, and test files use `test_*.py`. Prefer clear domain names such as `Face`, `Layer`, `Metal`, `Mesh`, feature lines, rails, and snap rules. Add type hints where they clarify public APIs or nontrivial geometry transformations.

No formatter or linter is configured in `pyproject.toml`; keep changes consistent with surrounding code and avoid unrelated style-only rewrites.

## Testing Guidelines

Tests use `pytest`. Add focused tests under `tests/` for behavior changes in meshing, snapping, feature-line extraction, classification, or dragging. Keep test inputs explicit and small enough that failures point to the broken algorithm step. Regression fixes should include a test that would fail without the change.

## Commit & Pull Request Guidelines

Git history uses short, direct commit summaries such as `fix drag` and `add data struct`. Keep that style, but make the affected area clear, for example `fix snap rule ordering` or `add box mesh regression test`.

Pull requests should include a concise description, the reason for the change, and the tests run. Link related issues when available. For visual or geometry-output changes, include screenshots or a short before/after note describing the mesh difference.

## Agent-Specific Instructions

Do not edit generated caches such as `__pycache__/` or `.pytest_cache/`. Keep new code inside the existing `src/`, `tests/`, `docs/`, or `script/` structure unless a new top-level file is explicitly required.
