"""Notebook kernel smoke test — §V3.

Extracts all code cells from notebooks/voila.ipynb and executes them as a
Python script in a subprocess.  This is intentionally NOT marked @pytest.mark.e2e
so it runs before Playwright is started, giving a fast, readable failure
instead of an opaque timeout when a Python-level bug (TraitError, ImportError,
SyntaxError, widget misconfiguration) prevents the notebook from serving.

No Jupyter kernel infrastructure is required — the subprocess uses the same
Python executable as the test runner.

Run inside the itisfoundation/jupyter-math container:
    /home/jovyan/.venv/bin/python -m pytest -v -o "addopts=" \
        -m notebook_smoke tests/test_notebook_smoke.py

Requires:
    - ipywidgets, IPython (available in the jupyter-math image)
    - git-lfs data pulled (data/database/ must contain at least one CSV)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NOTEBOOK = _REPO_ROOT / "notebooks" / "voila.ipynb"
_EXEC_TIMEOUT_S = 90


@pytest.mark.notebook_smoke
def test_v3_notebook_kernel_executes_without_error() -> None:
    """§V3: voila notebook cells must execute as Python without raising an exception."""
    nb = json.loads(_NOTEBOOK.read_text())
    code_cells = [
        "".join(cell["source"])
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code" and "".join(cell.get("source", []))
    ]
    source = "\n\n".join(code_cells)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="voila_smoke_"
    ) as tmp:
        tmp.write(source)
        tmp_path = tmp.name

    result = subprocess.run(
        [sys.executable, tmp_path],
        capture_output=True,
        text=True,
        timeout=_EXEC_TIMEOUT_S,
        cwd=str(_REPO_ROOT),
    )

    assert result.returncode == 0, (
        "Notebook code raised an exception — fix this before running Playwright.\n\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )
