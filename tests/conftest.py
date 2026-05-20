import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from .helpers import gaussian_2d, make_rect_grid, write_sar_csv

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SAVE_MEASUREMENT_VALIDATION_PLOTS", "0")
os.environ.setdefault("SAVE_TUTORIAL_VALIDATION_PLOTS", "0")

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _tail_text_file(path: Path, max_chars: int = 8000) -> str:
    """Return the tail of a UTF-8 text file for failure diagnostics."""
    if not path.exists():
        return f"<missing log file: {path}>"

    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow during broad test runs.",
    )
    parser.addoption(
        "--run-validation",
        action="store_true",
        default=False,
        help="Run tests marked as validation (requires Git LFS data).",
    )
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end Playwright tests (requires playwright browsers installed; run with -p no:xdist).",
    )


def _is_explicit_selection(config: pytest.Config) -> bool:
    # VS Code "Run All" and common bulk invocations target a top-level directory.
    # Keep those fast by default, but allow explicit file/test runs to include slow tests.
    args = [str(arg) for arg in config.args]
    if not args:
        return False

    for arg in args:
        if "::" in arg:
            return True
        if arg.endswith(".py"):
            return True

    return False


def _should_skip_slow(config: pytest.Config) -> bool:
    if bool(config.getoption("--run-slow")):
        return False
    if _is_explicit_selection(config):
        return False

    markexpr = str(getattr(config.option, "markexpr", "") or "").strip()
    return not (markexpr and "slow" in markexpr and "not slow" not in markexpr)


def _should_skip_validation(config: pytest.Config) -> bool:
    if bool(config.getoption("--run-validation")):
        return False
    if _is_explicit_selection(config):
        return False

    markexpr = str(getattr(config.option, "markexpr", "") or "").strip()
    return not (
        markexpr and "validation" in markexpr and "not validation" not in markexpr
    )


def _should_skip_e2e(config: pytest.Config) -> bool:
    if bool(config.getoption("--run-e2e")):
        return False
    return not _is_explicit_selection(config)


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("slow") and _should_skip_slow(item.config):
        pytest.skip(
            "slow test skipped by default; select it directly or pass --run-slow"
        )
    if item.get_closest_marker("validation") and _should_skip_validation(item.config):
        pytest.skip(
            "validation test skipped by default; select it directly or pass --run-validation"
        )
    if item.get_closest_marker("e2e") and _should_skip_e2e(item.config):
        pytest.skip(
            "e2e test skipped by default; pass --run-e2e and run with -p no:xdist"
        )


@pytest.fixture(autouse=True)
def disable_interactive_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    def fail_show(*args, **kwargs) -> None:
        raise AssertionError("Interactive matplotlib windows are disabled in tests")

    monkeypatch.setattr(plt, "show", fail_show)


@pytest.fixture
def tmp_csv_pair(tmp_path: Path):
    """
    Two small, identical rectangular-lattice CSVs.

    Convention in the pipeline:
      - measured CSV -> fixed image (target space)
      - reference CSV -> moving image (warped onto measured)
    """
    x, y = make_rect_grid(step=0.006)
    _, _, Z = gaussian_2d(x, y, x0=0.01, y0=-0.01, sx=0.02, sy=0.03, peak=1.0)

    measured = tmp_path / "measured.csv"
    reference = tmp_path / "reference.csv"

    write_sar_csv(measured, x, y, Z)
    write_sar_csv(reference, x, y, Z)

    return str(measured), str(reference)


# ---------------------------------------------------------------------------
# Voila server fixture for Playwright e2e tests
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def voila_server(tmp_path_factory: pytest.TempPathFactory):
    """Spawns a real voila server against a temp workspace mirroring the cloud layout.

    Yields ``(base_url, workspace_root)``. The notebook is copied to
    ``workspace_root/voila.ipynb`` and a ``sar-pattern-validation`` symlink
    points at the repo root, matching what osparc_makefile/Makefile sets up
    inside the container.
    """
    notebook_source = _REPO_ROOT / "notebooks" / "voila.ipynb"
    port = _free_port()
    workspace_root = tmp_path_factory.mktemp("voila-e2e-workspace")
    voila_log_path = workspace_root / "voila.log"
    (workspace_root / "sar-pattern-validation").symlink_to(_REPO_ROOT)
    shutil.copy2(notebook_source, workspace_root / "voila.ipynb")

    env = os.environ.copy()
    env["SAR_PATTERN_VALIDATION_BACKEND_MODE"] = "local"
    env["SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE"] = str(_REPO_ROOT)
    env["MPLBACKEND"] = "Agg"

    log_handle = voila_log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "voila",
                "voila.ipynb",
                "--no-browser",
                "--show_tracebacks=True",
                f"--port={port}",
                "--Voila.ip=127.0.0.1",
            ],
            cwd=workspace_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        base_url = f"http://127.0.0.1:{port}"
        deadline = time.time() + 120
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                log_handle.flush()
                out = _tail_text_file(voila_log_path)
                pytest.fail(f"Voila exited early (code {proc.returncode}):\n{out}")
            try:
                with urllib.request.urlopen(base_url + "/", timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(1)

        if not ready:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log_handle.flush()
            log_tail = _tail_text_file(voila_log_path)
            pytest.fail(f"Timed out waiting for Voila server to start.\n{log_tail}")

        yield base_url, workspace_root, voila_log_path

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        log_handle.close()
