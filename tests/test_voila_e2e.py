"""End-to-end Playwright tests for the Voila UI.

Run inside the jupyter-math container via:
    make test-voila-e2e

Manual repro inside `make serve-voila`, then:
    /home/jovyan/.venv/bin/python -m pytest -v -s -o "addopts=" \\
        --run-e2e -p no:xdist tests/test_voila_e2e.py::<name>

Design:
- All tests share one browser page to avoid restarting the voila kernel per
  test (heavy imports make each startup ~30-90s on WSL2).
- Tests are ordered from read-only to state-modifying; later tests deliberately
  build on earlier state.
- Every fixture and helper logs entry/exit + key state via `_log()` to stdout.
  Run with `-s` to see the trail; use it as a reproduction script when a test
  fails — each `>>` / `<<` line names the action taken or awaited.
- Tests requiring features not yet ported back into main-melanie carry a
  `pytest.mark.skip` with the gating phase; the skips are removed by the
  cherry-pick PRs that bring the underlying notebook feature in.

DOM notes (ipywidgets 8.x + voila 0.5):
- Toggle buttons: class="... widget-toggle-button"; .mod-active added on selection
- FileUpload: class="... widget-upload"; no <input type=file> in DOM —
  intercept via expect_file_chooser()
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from pathlib import Path

import pytest
from attr import dataclass

pytest.importorskip("playwright")

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KERNEL_TIMEOUT = 120_000  # ms — kernel startup + initial render
_UPLOAD_CSV_PATH = _REPO_ROOT / "data" / "example" / "measured_sSAR1g.csv"

# Reference selection used by _ensure_run_button_enabled. The four (column,
# value) pairs uniquely identify a row in data/database/, so the run button
# can enable deterministically. The resulting reference CSV is the file CI
# must pull from Git LFS — keep this in sync with
# .github/workflows/ci.yml's `git lfs pull`.
#   filename: dipole_1450MHz_Flat_5mm_1g.csv (smallest dipole 1g reference;
#   the previous blind-iteration code converged on the same row via sorted
#   order, so keeping it matches the workflow runtime the suite is tuned for)
_REFERENCE_FILTERS: tuple[tuple[str, str], ...] = (
    ("Antenna Type", "dipole"),
    ("Frequency [MHz]", "1450.0"),
    ("Distance [mm]", "5.0"),
    ("Mass [g]", "1.0"),
)
_REFERENCE_VALUES: frozenset[str] = frozenset(v for _, v in _REFERENCE_FILTERS)


# ---------------------------------------------------------------------------
# Logging helper — plain print() so `pytest -v -s` shows the trail.
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Print a timestamped trace line so each step is greppable in CI logs."""
    stamp = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
    print(f"[{stamp}] {msg}", flush=True)


def _tail_text_file(path: Path, max_chars: int = 8000) -> str:
    """Return the tail of a UTF-8 text file for failure diagnostics."""
    if not path.exists():
        return f"<missing log file: {path}>"

    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


# ---------------------------------------------------------------------------
# Shared page fixture — kernel starts once for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def voila_page(playwright, voila_server):
    """Navigate to voila once and keep the page alive for all tests."""
    base_url, _, voila_log_path = voila_server
    _log(f">> voila_page: launching headless chromium (server={base_url})")
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(_KERNEL_TIMEOUT)

    started = time.time()
    _log(f">> voila_page: page.goto({base_url}/)")
    page.goto(base_url + "/", timeout=_KERNEL_TIMEOUT)

    _log(
        f">> voila_page: waiting for selector .widget-button (timeout={_KERNEL_TIMEOUT}ms)"
    )
    try:
        page.wait_for_selector(".widget-button", timeout=_KERNEL_TIMEOUT)
    except Exception as exc:
        page_html = page.content()
        body_text = page.locator("body").inner_text(timeout=5_000)
        log_tail = _tail_text_file(voila_log_path)
        raise AssertionError(
            "Voila page loaded but no widgets rendered within the kernel timeout.\n\n"
            f"Body text:\n{body_text.strip() or '<empty body>'}\n\n"
            f"Page HTML:\n{page_html[:8000]}\n\n"
            f"Voila log tail ({voila_log_path}):\n{log_tail}"
        ) from exc
    _log(f"<< voila_page: kernel ready after {time.time() - started:.1f}s")

    yield page

    _log(">> voila_page: teardown — closing context + browser")
    context.close()
    browser.close()
    _log("<< voila_page: teardown complete")


@pytest.fixture(autouse=True)
def _capture_final_screenshot(request, voila_page):
    """Save a PNG of the final browser state after every test (pass and fail).

    Artifacts land in PLAYWRIGHT_ARTIFACTS_DIR (set by the test-harness script)
    or fall back to ``test-artifacts/playwright/`` relative to the repo root.
    File is named after the test function so it's unambiguous in CI and local
    review.
    """
    yield
    artifacts_dir = Path(
        os.environ.get(
            "PLAYWRIGHT_ARTIFACTS_DIR",
            str(_REPO_ROOT / "test-artifacts" / "playwright"),
        )
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w-]", "_", request.node.name)
    screenshot_path = artifacts_dir / f"{safe_name}.png"
    try:
        voila_page.screenshot(path=str(screenshot_path), full_page=True)
        _log(f"   screenshot → {screenshot_path}")
    except Exception as exc:  # noqa: BLE001
        _log(f"   screenshot failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers (each logs entry+exit so a hung helper is obvious in the trace)
# ---------------------------------------------------------------------------


def _upload_file(voila_page, file_path: Path) -> None:
    _log(f">> upload_file: clicking .widget-upload to upload {file_path.name}")
    with voila_page.expect_file_chooser(timeout=5_000) as fc:
        voila_page.locator(".widget-upload").click()
    fc.value.set_files(str(file_path))
    _log(f">> upload_file: waiting for {file_path.name!r} to appear in body text")
    voila_page.wait_for_function(
        "(expected) => document.body.innerText.includes(expected)",
        arg=file_path.name,
        timeout=15_000,
    )
    _log(f"<< upload_file: {file_path.name} visible in DOM")


def _ensure_run_button_enabled(voila_page) -> None:
    _log(">> ensure_run_button_enabled: starting")
    if _UPLOAD_CSV_PATH.name not in voila_page.locator("body").inner_text():
        _log("   no upload yet — performing initial upload")
        _upload_file(voila_page, _UPLOAD_CSV_PATH)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    _FIND_RUN_BTN = (
        "() => [...document.querySelectorAll('button')]"
        ".find(b => b.textContent.includes('Compare Patterns'))"
    )

    if run_btn.get_attribute("disabled") is None:
        _log("<< ensure_run_button_enabled: already enabled")
        return

    # Click the target value in each filter column. Each column is a VBox
    # whose first child is `<b>{column_name}</b>`. We anchor on the <b> and
    # take its *closest* widget-vbox ancestor — a plain `.widget-vbox:has(b…)`
    # would also match parent VBoxes (e.g. the cell-level wrapper) and pull
    # in toggles from neighbour columns like "10.0" appearing in both
    # Distance=10mm and Mass=10g. `:text-is()` keeps "1.0" from also
    # matching "10.0" inside the same column.
    for column_name, value in _REFERENCE_FILTERS:
        column = voila_page.locator(
            f'xpath=//b[normalize-space()="{column_name}"]'
            "/ancestor::*[contains(concat(' ', normalize-space(@class), ' '),"
            " ' widget-vbox ')][1]"
        )
        target_btn = column.locator(f'.widget-toggle-button:text-is("{value}")')
        classes = target_btn.get_attribute("class") or ""
        if "mod-active" in classes:
            _log(f"   {column_name} → {value} already active; skipping")
            continue
        _log(f"   clicking {column_name} → {value}")
        target_btn.click()
        voila_page.wait_for_timeout(200)

    # check_settings polls every 1 s; wait up to 3 s for it to re-enable.
    with contextlib.suppress(Exception):
        voila_page.wait_for_function(
            f"() => {{ const b = ({_FIND_RUN_BTN})(); return b && !b.disabled; }}",
            timeout=3_000,
        )

    if run_btn.get_attribute("disabled") is not None:
        # Defensive fallback: clear any stale toggles activated by earlier
        # tests that don't match our four target values, then re-wait.
        _log("   run still disabled — clearing stale toggles")
        active = voila_page.locator(".widget-toggle-button.mod-active")
        for i in range(active.count()):
            btn = active.nth(i)
            text = (btn.text_content() or "").strip()
            if text in _REFERENCE_VALUES:
                continue
            with contextlib.suppress(Exception):
                btn.click()
                voila_page.wait_for_timeout(200)
        with contextlib.suppress(Exception):
            voila_page.wait_for_function(
                f"() => {{ const b = ({_FIND_RUN_BTN})(); return b && !b.disabled; }}",
                timeout=3_000,
            )

    assert run_btn.get_attribute("disabled") is None, (
        "Run button should be enabled once a measured file and unique reference are selected"
    )
    _log("<< ensure_run_button_enabled: enabled")


def _wait_for_workflow_cycle(voila_page, timeout_ms: int = 300_000) -> None:
    """Wait for Compare Patterns to disable (running) then re-enable (done).
    Drives off the button-state cycle, not body text. When the page has stale
    result content in the DOM (restored session, prior run), any text-based
    terminal-state probe matches immediately and hides whether the new run
    actually completed. The disable→enable transition is the only signal that
    survives reruns.
    """
    _log(">> wait_for_workflow_cycle: starting")
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    _FIND_BTN = (
        "() => [...document.querySelectorAll('button')]"
        ".find(b => b.textContent.includes('Compare Patterns'))"
    )
    _log("   waiting up to 10s for run button to become disabled (cycle started)")
    voila_page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN})(); return b && b.disabled; }}",
        timeout=10_000,
    )

    _log(
        f"   waiting up to {timeout_ms / 1000:.0f}s for run button to re-enable (cycle done)"
    )
    voila_page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN})(); return b && !b.disabled; }}",
        timeout=timeout_ms,
    )
    assert run_btn.get_attribute("disabled") is None

    # The notebook's handle_button_click renders the result table *before* the
    # `finally` branch re-enables the button, so by this point the DOM should
    # already be in a terminal state. Verify it explicitly so a silent error
    # banner (`Error: …` from _set_feedback_banner) fails the test rather than
    # being masked by stale "Pass rate" text from an earlier run.
    terminal_state_js = (
        "() => {"
        "  const bodyText = document.body.innerText;"
        "  const bodyHtml = document.body.innerHTML;"
        "  return bodyText.includes('Pass rate')"
        "    || bodyHtml.includes('Reference, 30 dBm')"
        "    || bodyText.includes('already match the current results')"
        "    || bodyText.includes('SAR pattern validation complete.')"
        "    || bodyText.includes('Warning:')"
        "    || bodyText.includes('Could not reach the Voila server')"
        "    || bodyText.includes('Workflow execution failed')"
        "    || /\\bError:\\s/.test(bodyText);"
        "}"
    )
    _log("   verifying rendered terminal state in DOM")
    voila_page.wait_for_function(
        terminal_state_js,
        timeout=10_000,
    )
    _log("<< wait_for_workflow_cycle: complete")


@dataclass
class PSSARRowValues:
    measured_value: float
    measured_30dbm: float
    reference_value: float
    scaling_error: float


def _extract_pssar_row_values(page_html: str) -> PSSARRowValues:
    _log(">> extract_pssar_row_values: scanning page HTML")
    # New two-table layout: result badge | measured@power | measured@30dBm | reference@30dBm | scaling_error | criteria
    match = re.search(
        r"Reference, 30 dBm</th>.*?"  # anchor on header
        r"(?:Pass|Fail).*?</td>"  # skip result badge cell
        r"\s*<td[^>]*>[\d.]+\s*W/kg</td>"  # skip measured@input_power
        r"\s*<td[^>]*>([\d.]+)\s*W/kg</td>"  # measured@30dBm
        r"\s*<td[^>]*>([\d.]+)\s*W/kg</td>"  # reference@30dBm
        r"\s*<td[^>]*>([-\d.]+)</td>",  # scaling error [%]
        page_html,
        flags=re.S,
    )
    assert match is not None, "Could not extract result table values from page."
    values = PSSARRowValues(
        measured_value=float(match.group(1)),  # measured 30 dBm
        measured_30dbm=float(match.group(1)),
        reference_value=float(match.group(2)),  # reference 30 dBm
        scaling_error=float(match.group(3)),  # scaling error [%]
    )
    _log(f"<< extract_pssar_row_values: {values}")
    return values


def _set_power_level(voila_page, value: float) -> None:
    _log(f">> set_power_level: setting power input to {value}")
    power_input = voila_page.locator("input[type='number']").first
    power_input.click()
    power_input.fill(str(value))
    power_input.press("Tab")
    voila_page.wait_for_function(
        "(expected) => {"
        "  const input = document.querySelector(\"input[type='number']\");"
        "  return input && Math.abs(Number(input.value) - expected) < 0.01;"
        "}",
        arg=value,
        timeout=10_000,
    )
    _log(f"<< set_power_level: confirmed value={value}")


# ---------------------------------------------------------------------------
# Read-only smoke tests (run first — page is in fresh state)
# ---------------------------------------------------------------------------


def test_run_button_is_visible(voila_page) -> None:
    _log(">> test_run_button_is_visible")
    assert voila_page.locator("button:has-text('Compare Patterns')").is_visible()
    _log("<< test_run_button_is_visible: pass")


def test_run_button_is_disabled_on_fresh_load(voila_page) -> None:
    _log(">> test_run_button_is_disabled_on_fresh_load")
    btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert btn.get_attribute("disabled") is not None
    _log("<< test_run_button_is_disabled_on_fresh_load: pass")


def test_filter_toggle_buttons_are_visible(voila_page) -> None:
    _log(">> test_filter_toggle_buttons_are_visible")
    count = voila_page.locator(".widget-toggle-button").count()
    _log(f"   toggle button count = {count}")
    assert count > 0
    _log("<< test_filter_toggle_buttons_are_visible: pass")


def test_file_upload_button_is_present(voila_page) -> None:
    _log(">> test_file_upload_button_is_present")
    count = voila_page.locator(".widget-upload").count()
    _log(f"   widget-upload count = {count}")
    assert count == 1
    _log("<< test_file_upload_button_is_present: pass")


# ---------------------------------------------------------------------------
# State-modifying tests (each builds on the previous)
# ---------------------------------------------------------------------------


def test_clicking_filter_button_activates_it(voila_page) -> None:
    _log(">> test_clicking_filter_button_activates_it")
    voila_page.locator(".widget-toggle-button").first.click()
    voila_page.wait_for_selector(".widget-toggle-button.mod-active", timeout=10_000)
    active_count = voila_page.locator(".widget-toggle-button.mod-active").count()
    _log(f"   active toggle count after click = {active_count}")
    assert active_count >= 1
    _log("<< test_clicking_filter_button_activates_it: pass")


def test_file_upload_updates_filename_label(voila_page) -> None:
    _log(">> test_file_upload_updates_filename_label")
    _upload_file(voila_page, _UPLOAD_CSV_PATH)
    _log("<< test_file_upload_updates_filename_label: pass")


def test_run_button_enables_after_upload_and_unique_filter(voila_page) -> None:
    """Clicks filter buttons until exactly one reference matches; asserts run is enabled."""
    _log(">> test_run_button_enables_after_upload_and_unique_filter")
    _ensure_run_button_enabled(voila_page)
    _log("<< test_run_button_enables_after_upload_and_unique_filter: pass")


# ---------------------------------------------------------------------------
# Tests gated on features that haven't been cherry-picked back into main-melanie.
# Each Phase B PR removes the matching skip mark when it lands the feature.
# ---------------------------------------------------------------------------


def test_run_workflow_and_check_results_table(voila_page) -> None:
    """Clicks Compare Patterns and asserts the results tables render without error."""
    _log(">> test_run_workflow_and_check_results_table")
    _ensure_run_button_enabled(voila_page)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert run_btn.get_attribute("disabled") is None, "Run button must be enabled first"

    _log("   clicking Compare Patterns")
    run_btn.click()
    _wait_for_workflow_cycle(voila_page)

    body_text = voila_page.locator("body").inner_text()
    page_html = voila_page.content()

    assert "Traceback" not in body_text, (
        f"Python traceback in page:\n{body_text[:3000]}"
    )
    assert not re.search(r"\bError:\s", body_text), (
        f"Error banner in page after workflow ran:\n{body_text[:3000]}"
    )
    assert "Workflow execution failed" not in body_text, (
        f"Workflow failure banner in page:\n{body_text[:3000]}"
    )
    assert "Reference, 30 dBm" in page_html, (
        f"Result table not found.\nBody text:\n{body_text[:3000]}\n\nPage HTML tail:\n{page_html[-2000:]}"
    )
    assert "Pass rate" in body_text, (
        f"Pass rate not found.\nBody text:\n{body_text[:3000]}"
    )
    assert "Pass" in body_text or "Fail" in body_text, (
        f"No Pass/Fail result found.\nBody text:\n{body_text[:3000]}"
    )
    _log("<< test_run_workflow_and_check_results_table: pass")


def test_restored_session_rerun_updates_results_after_power_change(voila_page) -> None:
    _log(">> test_restored_session_rerun_updates_results_after_power_change")
    _ensure_run_button_enabled(voila_page)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    if "Reference, 30 dBm" not in voila_page.content():
        _log("   no prior results in DOM — running once to seed state")
        run_btn.click()
        _wait_for_workflow_cycle(voila_page)
    first_values = _extract_pssar_row_values(voila_page.content())

    _log(f"   reloading page (timeout={_KERNEL_TIMEOUT}ms)")
    voila_page.reload(timeout=_KERNEL_TIMEOUT)
    voila_page.wait_for_selector(".widget-button", timeout=_KERNEL_TIMEOUT)
    voila_page.wait_for_function(
        f"() => document.body.innerText.includes('{_UPLOAD_CSV_PATH.name}')",
        timeout=15_000,
    )
    voila_page.wait_for_function(
        "() => document.body.innerText.includes('Reference, 30 dBm')",
        timeout=15_000,
    )

    restored_values = _extract_pssar_row_values(voila_page.content())
    assert restored_values.measured_30dbm == pytest.approx(
        first_values.measured_30dbm, abs=0.01
    )
    assert restored_values.reference_value == pytest.approx(
        first_values.reference_value, abs=0.01
    )
    assert restored_values.scaling_error == pytest.approx(
        first_values.scaling_error, abs=0.01
    )

    _set_power_level(voila_page, 10.0)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    # check_settings re-enables the button within ~1s after restore; wait up to 3s.
    _FIND_RUN_BTN = (
        "() => [...document.querySelectorAll('button')]"
        ".find(b => b.textContent.includes('Compare Patterns'))"
    )
    with contextlib.suppress(Exception):
        voila_page.wait_for_function(
            f"() => {{ const b = ({_FIND_RUN_BTN})(); return b && !b.disabled; }}",
            timeout=3_000,
        )
    assert run_btn.get_attribute("disabled") is None

    _log("   clicking Compare Patterns after power change")
    run_btn.click()
    _wait_for_workflow_cycle(voila_page)

    second_values = _extract_pssar_row_values(voila_page.content())
    # Verify the run completed and results are displayed (not a memo-cache early return).
    # Note: measured_pssar is not scaled by power_level_dbm in the current CLI, so we
    # assert results are *present* and consistent with using the same reference file.
    assert second_values.reference_value == pytest.approx(
        first_values.reference_value, abs=0.01
    ), "Reference pssar should be unchanged (same reference file used)"
    assert "Pass rate" in voila_page.locator("body").inner_text()
    assert (
        "already match the current results"
        not in voila_page.locator("body").inner_text()
    ), "Memo cache should NOT have fired — power level changed"
    _log("<< test_restored_session_rerun_updates_results_after_power_change: pass")


def test_same_session_rerun_updates_results_after_power_change(voila_page) -> None:
    _log(">> test_same_session_rerun_updates_results_after_power_change")
    _ensure_run_button_enabled(voila_page)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert run_btn.get_attribute("disabled") is None

    first_values = _extract_pssar_row_values(voila_page.content())

    _set_power_level(voila_page, 17.0)
    _log("   clicking Compare Patterns after power change")
    run_btn.click()
    # Power-level-only change takes the fast path (V6): no registration re-run,
    # no button cycle. Wait for the unique fast-path banner instead.
    _log("   waiting for fast-path 'Power level updated' banner")
    voila_page.wait_for_function(
        "() => document.body.innerText.includes('Power level updated')",
        timeout=10_000,
    )
    _log("   fast-path banner detected")

    second_values = _extract_pssar_row_values(voila_page.content())
    assert run_btn.get_attribute("disabled") is None
    # Verify results are present and not a memo-cache early return.
    assert second_values.reference_value == pytest.approx(
        first_values.reference_value, abs=0.01
    ), "Reference pssar should be unchanged (same reference file used)"
    assert "Pass rate" in voila_page.locator("body").inner_text()
    assert (
        "already match the current results"
        not in voila_page.locator("body").inner_text()
    ), "Memo cache should NOT have fired — power level changed from previous run"
    _log("<< test_same_session_rerun_updates_results_after_power_change: pass")


def test_exact_repeat_shows_warning_without_rerunning(voila_page) -> None:
    _log(">> test_exact_repeat_shows_warning_without_rerunning")
    _ensure_run_button_enabled(voila_page)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    previous_html = voila_page.content()

    _log("   clicking Compare Patterns to repeat an unchanged run")
    run_btn.click()
    voila_page.wait_for_function(
        "() => document.body.innerText.includes('already match the current results')",
        timeout=15_000,
    )

    assert run_btn.get_attribute("disabled") is None
    assert "Reference, 30 dBm" in previous_html
    assert "Reference, 30 dBm" in voila_page.content()
    _log("<< test_exact_repeat_shows_warning_without_rerunning: pass")


def test_uploading_new_data_clears_prior_results(voila_page, tmp_path: Path) -> None:
    _log(">> test_uploading_new_data_clears_prior_results")
    replacement_csv = tmp_path / "replacement_measured.csv"
    replacement_csv.write_text("x,y,sar\n0,0,2\n1,1,3\n", encoding="utf-8")

    assert "Reference, 30 dBm" in voila_page.content()
    _upload_file(voila_page, replacement_csv)
    _log("   waiting for result table to disappear from DOM")
    voila_page.wait_for_function(
        "() => !document.body.innerHTML.includes('Reference, 30 dBm')",
        timeout=10_000,
    )

    page_html = voila_page.content()
    assert "Reference, 30 dBm" not in page_html
    assert "psSAR" not in page_html
    _log("<< test_uploading_new_data_clears_prior_results: pass")


# ---------------------------------------------------------------------------
# Measurement-area input tests
# ---------------------------------------------------------------------------

_MEAS_X_DEFAULT = 0.0
_MEAS_Y_DEFAULT = 0.0
_MEAS_Y_MIN = 50.01

_FAST_TRACK_PASS_POWER_DBM = 21.0


def _meas_area_input(voila_page, label_fragment: str):
    """Return the number input for a measurement-area widget identified by label text."""
    return (
        voila_page.locator(".widget-text")
        .filter(has=voila_page.locator(f"label:has-text('{label_fragment}')"))
        .locator("input[type='number']")
    )


def _set_meas_area(voila_page, x: int, y: int) -> None:
    """Set measurement area x and y inputs via label-anchored locators.

    Uses label text instead of positional nth() so that adding widgets to the
    same row does not silently break this helper (§V5).
    """
    _log(f">> set_meas_area: x={x}, y={y}")
    for label_fragment, value in [("Meas. area x", x), ("Meas. area y", y)]:
        inp = _meas_area_input(voila_page, label_fragment)
        inp.click(click_count=3)
        inp.type(str(value))
        inp.press("Tab")
        voila_page.wait_for_function(
            "({label, expected}) => {"
            "  for (const c of document.querySelectorAll('.widget-text')) {"
            "    const lbl = c.querySelector('label');"
            "    const inp = c.querySelector('input[type=number]');"
            "    if (lbl && lbl.textContent.includes(label) && inp)"
            "      return Number(inp.value) === expected;"
            "  }"
            "  return false;"
            "}",
            arg={"label": label_fragment, "expected": value},
            timeout=5_000,
        )
    _log("<< set_meas_area: done")


def _click_run_expect_error(voila_page, error_fragment: str) -> None:
    """Click Run and wait for an error banner containing error_fragment."""
    _log(f">> click_run_expect_error: expecting {error_fragment!r}")
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert run_btn.get_attribute("disabled") is None, "Run button must be enabled"
    run_btn.click()
    voila_page.wait_for_function(
        "(text) => document.body.innerText.includes(text)",
        arg=error_fragment,
        timeout=10_000,
    )
    _log(f"<< click_run_expect_error: found {error_fragment!r}")


class TestMeasurementAreaInputs:
    def test_measurement_area_between_1_and_49_shows_error(self, voila_page) -> None:
        _log(">> test_measurement_area_between_1_and_49_shows_error")
        _ensure_run_button_enabled(voila_page)
        _set_meas_area(voila_page, 30, 30)
        _click_run_expect_error(voila_page, "Measurement area must be >= 50 mm")
        _log("<< test_measurement_area_between_1_and_49_shows_error: pass")

    def test_measurement_area_exactly_49_shows_error(self, voila_page) -> None:
        _log(">> test_measurement_area_exactly_49_shows_error")
        _ensure_run_button_enabled(voila_page)
        _set_meas_area(voila_page, 49, 49)
        _click_run_expect_error(voila_page, "Measurement area must be >= 50 mm")
        _log("<< test_measurement_area_exactly_49_shows_error: pass")

    def test_measurement_area_y_below_50_shows_error(self, voila_page) -> None:
        _log(">> test_measurement_area_y_below_50_shows_error")
        _ensure_run_button_enabled(voila_page)
        _set_meas_area(voila_page, 100, 30)
        _click_run_expect_error(voila_page, "Measurement area must be >= 50 mm")
        _log("<< test_measurement_area_y_below_50_shows_error: pass")

    def test_measurement_area_x_accepts_upper_bound_600(self, voila_page) -> None:
        _log(">> test_measurement_area_x_accepts_upper_bound_600")
        x_input = _meas_area_input(voila_page, "Meas. area x")
        x_input.click(click_count=3)
        x_input.type("600")
        x_input.press("Tab")
        voila_page.wait_for_function(
            "({label, expected}) => {"
            "  for (const c of document.querySelectorAll('.widget-text')) {"
            "    const lbl = c.querySelector('label');"
            "    const inp = c.querySelector('input[type=number]');"
            "    if (lbl && lbl.textContent.includes(label) && inp)"
            "      return Math.abs(Number(inp.value) - expected) < 0.01;"
            "  }"
            "  return false;"
            "}",
            arg={"label": "Meas. area x", "expected": 600},
            timeout=5_000,
        )
        val = float(x_input.input_value())
        assert abs(val - 600.0) < 0.01, f"Expected 600, got {val}"
        _log("<< test_measurement_area_x_accepts_upper_bound_600: pass")

    def test_measurement_area_y_accepts_upper_bound_400(self, voila_page) -> None:
        _log(">> test_measurement_area_y_accepts_upper_bound_400")
        y_input = _meas_area_input(voila_page, "Meas. area y")
        y_input.click(click_count=3)
        y_input.type("400")
        y_input.press("Tab")
        voila_page.wait_for_function(
            "({label, expected}) => {"
            "  for (const c of document.querySelectorAll('.widget-text')) {"
            "    const lbl = c.querySelector('label');"
            "    const inp = c.querySelector('input[type=number]');"
            "    if (lbl && lbl.textContent.includes(label) && inp)"
            "      return Math.abs(Number(inp.value) - expected) < 0.01;"
            "  }"
            "  return false;"
            "}",
            arg={"label": "Meas. area y", "expected": 400},
            timeout=5_000,
        )
        val = float(y_input.input_value())
        assert abs(val - 400.0) < 0.01, f"Expected 400, got {val}"
        _log("<< test_measurement_area_y_accepts_upper_bound_400: pass")

    def test_measurement_area_zero_auto_allows_run(self, voila_page) -> None:
        _log(">> test_measurement_area_zero_auto_allows_run")
        _ensure_run_button_enabled(voila_page)
        _set_meas_area(voila_page, 0, 0)
        run_btn = voila_page.locator("button:has-text('Compare Patterns')")
        assert run_btn.get_attribute("disabled") is None
        run_btn.click()
        _wait_for_workflow_cycle(voila_page)
        assert (
            "Measurement area must be >= 50 mm"
            not in voila_page.locator("body").inner_text()
        )
        _log("<< test_measurement_area_zero_auto_allows_run: pass")

    def test_measurement_area_restored_to_valid_before_run(self, voila_page) -> None:
        _log(">> test_measurement_area_restored_to_valid_before_run")
        _set_meas_area(voila_page, 300, 200)
        _log("<< test_measurement_area_restored_to_valid_before_run: pass")


def test_workflow_produces_square_plots(voila_page, voila_server) -> None:
    from PIL import Image

    _log(">> test_workflow_produces_square_plots")
    _, workspace_root, _ = voila_server
    img_path = workspace_root / "images" / "gamma_comparison_image.png"
    assert img_path.exists(), f"Output image not found at {img_path}"
    with Image.open(img_path) as img:
        w, h = img.size
    assert w == h, f"Expected square plot, got {w}×{h}"
    _log(f"<< test_workflow_produces_square_plots: pass (size={w}×{h})")


# ---------------------------------------------------------------------------
# Noise floor widget tests
# ---------------------------------------------------------------------------

_NOISE_FLOOR_LABEL_TEXT = "noise floor"
_NOISE_FLOOR_DEFAULT = 0.05
_NOISE_FLOOR_MAX = 0.05


def _noise_floor_input(voila_page):
    return (
        voila_page.locator(".widget-text")
        .filter(has=voila_page.locator(f"label:has-text('{_NOISE_FLOOR_LABEL_TEXT}')"))
        .locator("input[type='number']")
    )


def _set_noise_floor(voila_page, value: float) -> None:
    _log(f">> set_noise_floor: setting noise floor input to {value}")
    inp = _noise_floor_input(voila_page)
    inp.click()
    inp.fill(str(value))
    inp.press("Tab")
    voila_page.wait_for_timeout(300)
    _log(f"<< set_noise_floor: done (value={value})")


def test_noise_floor_widget_is_visible(voila_page) -> None:
    """The noise floor input widget must be visible on page load."""
    _log(">> test_noise_floor_widget_is_visible")
    assert _noise_floor_input(voila_page).is_visible()
    _log("<< test_noise_floor_widget_is_visible: pass")


def test_noise_floor_widget_default_value(voila_page) -> None:
    """The noise floor widget must default to 0.05 W/kg."""
    _log(">> test_noise_floor_widget_default_value")
    inp = _noise_floor_input(voila_page)
    actual = float(inp.input_value())
    _log(f"   noise_floor input value = {actual}")
    assert actual == pytest.approx(_NOISE_FLOOR_DEFAULT, abs=1e-9)
    _log("<< test_noise_floor_widget_default_value: pass")


def test_noise_floor_widget_clamped_at_max(voila_page) -> None:
    """Setting noise floor above 0.05 W/kg must be clamped to 0.05 (BoundedFloatText)."""
    _log(">> test_noise_floor_widget_clamped_at_max")
    _set_noise_floor(voila_page, 0.5)
    inp = _noise_floor_input(voila_page)
    actual = float(inp.input_value())
    _log(f"   noise_floor after setting 0.5 = {actual}")
    assert actual == pytest.approx(_NOISE_FLOOR_MAX, abs=1e-9)
    _set_noise_floor(voila_page, _NOISE_FLOOR_DEFAULT)
    _log("<< test_noise_floor_widget_clamped_at_max: pass")


def test_noise_floor_persisted_after_change_and_reload(voila_page) -> None:
    """Changing noise floor then reloading the page must restore the saved value."""
    _log(">> test_noise_floor_persisted_after_change_and_reload")
    target = 0.03
    _set_noise_floor(voila_page, target)

    _log(f"   reloading page (timeout={_KERNEL_TIMEOUT}ms)")
    voila_page.reload(timeout=_KERNEL_TIMEOUT)
    voila_page.wait_for_selector(".widget-button", timeout=_KERNEL_TIMEOUT)

    inp = _noise_floor_input(voila_page)
    actual = float(inp.input_value())
    _log(f"   noise_floor after reload = {actual}")
    assert actual == pytest.approx(target, abs=1e-9)

    _set_noise_floor(voila_page, _NOISE_FLOOR_DEFAULT)


def test_mask_too_small_shows_error_banner(voila_page, tmp_path) -> None:
    """Uploading a < 22 mm × 22 mm measured file must show a MASK_TOO_SMALL error banner."""
    import numpy as np
    import pandas as pd

    _log(">> test_mask_too_small_shows_error_banner")

    # Generate a 15 mm × 15 mm Gaussian SAR grid — smaller than the 22 mm inscribed-square threshold.
    step = 0.001
    xs = np.arange(-0.0075, 0.0076, step)
    ys = np.arange(-0.0075, 0.0076, step)
    X, Y = np.meshgrid(xs, ys)
    Z = 2.5 * np.exp(-((X**2 + Y**2) / (2 * 0.003**2)))
    tiny_csv = tmp_path / "tiny_measured_sSAR_15mm.csv"
    pd.DataFrame({"x [m]": X.ravel(), "y [m]": Y.ravel(), "SAR": Z.ravel()}).to_csv(
        tiny_csv, index=False
    )

    _ensure_run_button_enabled(voila_page)
    _log("   uploading tiny 15 mm × 15 mm measured CSV")
    _upload_file(voila_page, tiny_csv)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    _log("   clicking Compare Patterns")
    run_btn.click()
    _wait_for_workflow_cycle(voila_page, timeout_ms=120_000)

    body_text = voila_page.locator("body").inner_text()
    _log(f"   body snippet: {body_text[:300]!r}")
    assert "Error:" in body_text, "Expected an Error banner for MASK_TOO_SMALL"
    assert "22 mm" in body_text, "Expected '22 mm' in MASK_TOO_SMALL error text"

    _log("<< test_mask_too_small_shows_error_banner: pass")


def test_result_table_clears_on_rerun_then_repopulates(voila_page) -> None:
    """Result table must disappear when a new run starts and reappear when done.

    Bug: on the second Compare Patterns click the images are cleared but the
    result table keeps showing stale data from the previous run.  Fix: set
    result_table.value = "" at the top of handle_button_click, before
    update_images(no_data=True).
    """
    _log(">> test_result_table_clears_on_rerun_then_repopulates")

    # Restore valid CSV so the run can succeed.
    if _UPLOAD_CSV_PATH.name not in voila_page.locator("body").inner_text():
        _upload_file(voila_page, _UPLOAD_CSV_PATH)
    _ensure_run_button_enabled(voila_page)
    _set_meas_area(voila_page, 0, 0)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")

    # First run: produce a result table with content.
    run_btn.click()
    _wait_for_workflow_cycle(voila_page)
    first_html = voila_page.content()
    assert "Reference, 30 dBm" in first_html, (
        "Prerequisite: first run must produce a result table"
    )
    _log("   first run complete — result table confirmed")

    # Force a different run key so the second click triggers a full rerun
    # (not the 'already match' early return).  Toggling noise_floor briefly
    # changes the key and restores it so the test leaves the page clean.
    # Use 0.03: the widget max is 0.05 so 0.06 would clamp to 0.05 (= same key).
    _set_noise_floor(voila_page, 0.03)

    # Second run: click and immediately check the table clears while running.
    _log("   clicking Compare Patterns for second run")
    run_btn.click()

    _FIND_BTN = (
        "() => [...document.querySelectorAll('button')]"
        ".find(b => b.textContent.includes('Compare Patterns'))"
    )
    voila_page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN})(); return b && b.disabled; }}",
        timeout=10_000,
    )
    _log("   run started — asserting result table is now empty")
    table_html = voila_page.locator("body").inner_html()
    assert "Reference, 30 dBm" not in table_html, (
        "Result table must be cleared when a rerun starts "
        "(found stale 'Reference, 30 dBm' header while button was disabled)"
    )

    # Wait for run to finish and confirm table repopulates.
    _wait_for_workflow_cycle(voila_page)
    second_html = voila_page.content()
    assert "Reference, 30 dBm" in second_html, (
        "Result table must repopulate after the rerun completes"
    )
    _log("   result table repopulated after second run")

    # Restore noise floor for subsequent tests.
    _set_noise_floor(voila_page, _NOISE_FLOOR_DEFAULT)
    _log("<< test_result_table_clears_on_rerun_then_repopulates: pass")


# ---------------------------------------------------------------------------
# V12 / B11: fast-track power-level rescale must update Measured@30dBm and
# Scaling Error, not just the first column. Both tests below are E2E gates
# for §V12 and will FAIL until the B11 fix lands in voila.ipynb.
# ---------------------------------------------------------------------------


def _extract_pssar_result(page_html: str) -> str:
    """Return 'Pass' or 'Fail' for the psSAR result badge in the result table."""
    match = re.search(
        r"Reference, 30 dBm</th>.*?(Pass|Fail)",
        page_html,
        flags=re.S,
    )
    assert match is not None, "Could not find psSAR Pass/Fail badge in page HTML"
    return match.group(1)


def test_fast_track_wrong_power_after_success_shows_failure(voila_page) -> None:
    """V12 / B11: wrong power via fast-track must update Measured@30dBm and Scaling Error → Fail.

    At 1 dBm the normalization factor is 10^(20/10) = 100× larger than at 21 dBm,
    so Measured@30dBm balloons and Scaling Error far exceeds ±10 % → psSAR Fail.
    The bug: _update_analytical_results receives stale workflow_results (computed at
    21 dBm) so the second and third columns are frozen and the badge stays Pass.
    """
    _log(">> test_fast_track_wrong_power_after_success_shows_failure")

    # Upload the original valid CSV (prior test left a tiny CSV — this forces a
    # full workflow run, not a fast-track, for the 28 dBm baseline below).
    if _UPLOAD_CSV_PATH.name not in voila_page.locator("body").inner_text():
        _upload_file(voila_page, _UPLOAD_CSV_PATH)
    _ensure_run_button_enabled(voila_page)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")

    # Baseline: full run at 28 dBm (data is calibrated for this power).
    _set_power_level(voila_page, _FAST_TRACK_PASS_POWER_DBM)
    _log(f"   baseline full run at {_FAST_TRACK_PASS_POWER_DBM} dBm")
    run_btn.click()
    _wait_for_workflow_cycle(voila_page)

    baseline = _extract_pssar_row_values(voila_page.content())
    _log(
        f"   baseline: measured_30dbm={baseline.measured_30dbm:.3f} W/kg, "
        f"scaling_error={baseline.scaling_error:.1f}%"
    )
    assert abs(baseline.scaling_error) <= 25.0, (
        f"Baseline run at {_FAST_TRACK_PASS_POWER_DBM} dBm must pass psSAR "
        f"(scaling_error={baseline.scaling_error:.1f}%). "
        "If the example data requires a different power, update _FAST_TRACK_PASS_POWER_DBM."
    )

    # Fast-track at 1 dBm (20 dBm below baseline → Measured@30dBm grows 100×).
    _set_power_level(voila_page, 1.0)
    _log("   fast-track at wrong power 1 dBm")
    run_btn.click()
    voila_page.wait_for_function(
        "() => document.body.innerText.includes('Power level updated')",
        timeout=10_000,
    )
    _log("   'Power level updated' banner confirmed")

    after = _extract_pssar_row_values(voila_page.content())
    _log(
        f"   after wrong power: measured_30dbm={after.measured_30dbm:.3f} W/kg, "
        f"scaling_error={after.scaling_error:.1f}%"
    )

    # V12: Measured@30dBm must increase substantially (bug: it stays frozen).
    assert after.measured_30dbm > baseline.measured_30dbm * 10, (
        f"Measured@30dBm must grow when power drops {_FAST_TRACK_PASS_POWER_DBM}→1 dBm via fast-track "
        f"(before={baseline.measured_30dbm:.3f}, after={after.measured_30dbm:.3f})"
    )
    # V12: Scaling error must now far exceed ±25 % → Fail.
    assert abs(after.scaling_error) > 25.0, (
        f"Scaling error must exceed ±25 % at wrong power 1 dBm "
        f"(got {after.scaling_error:.1f}%)"
    )
    assert _extract_pssar_result(voila_page.content()) == "Fail", (
        "psSAR badge must be Fail when power level change causes huge scaling error"
    )
    _log("<< test_fast_track_wrong_power_after_success_shows_failure: pass")


def test_fast_track_fix_power_restores_pass(voila_page) -> None:
    """V12 / B11: correcting power level via fast-track must restore psSAR Pass.

    Sequence: full run at 1 dBm (wrong) → Fail; fast-track to 21 dBm (correct) → Pass.
    To force the wrong-power run to be a full workflow (not a fast-track from the
    previous test), we use a slightly different noise_floor (0.06 instead of 0.05)
    so the run key does not match.  After asserting the fix we restore noise_floor.
    The bug: fast-track keeps stale scaling_error so the badge stays Fail even after
    the correct power is entered.
    """
    _log(">> test_fast_track_fix_power_restores_pass")

    if _UPLOAD_CSV_PATH.name not in voila_page.locator("body").inner_text():
        _upload_file(voila_page, _UPLOAD_CSV_PATH)
    _ensure_run_button_enabled(voila_page)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")

    # Full run at wrong power using noise_floor=0.03 so the run key differs from
    # any prior fast-track entry (which used 0.05) — guarantees a full workflow.
    # Use 0.03, not 0.06: the widget max is 0.05 so 0.06 would clamp to 0.05 (= same key).
    _set_noise_floor(voila_page, 0.03)
    _set_power_level(voila_page, 1.0)
    _log("   full run at wrong power 1 dBm (noise_floor=0.03)")
    run_btn.click()
    _wait_for_workflow_cycle(voila_page)

    wrong = _extract_pssar_row_values(voila_page.content())
    _log(
        f"   wrong power run: measured_30dbm={wrong.measured_30dbm:.3f} W/kg, "
        f"scaling_error={wrong.scaling_error:.1f}%"
    )
    assert abs(wrong.scaling_error) > 25.0, (
        f"Run at 1 dBm must fail psSAR (scaling_error={wrong.scaling_error:.1f}%)"
    )
    assert _extract_pssar_result(voila_page.content()) == "Fail", (
        "psSAR badge must be Fail at wrong power 1 dBm"
    )

    # Fast-track to correct power (noise_floor stays 0.06 — only power changes).
    _set_power_level(voila_page, _FAST_TRACK_PASS_POWER_DBM)
    _log(f"   fast-track to correct power {_FAST_TRACK_PASS_POWER_DBM} dBm")
    run_btn.click()
    voila_page.wait_for_function(
        "() => document.body.innerText.includes('Power level updated')",
        timeout=10_000,
    )
    _log("   'Power level updated' banner confirmed")

    fixed = _extract_pssar_row_values(voila_page.content())
    _log(
        f"   after fix: measured_30dbm={fixed.measured_30dbm:.3f} W/kg, "
        f"scaling_error={fixed.scaling_error:.1f}%"
    )

    # V12: Measured@30dBm must shrink substantially (bug: it stays frozen at huge value).
    assert fixed.measured_30dbm < wrong.measured_30dbm / 10, (
        f"Measured@30dBm must decrease when power rises from 1→{_FAST_TRACK_PASS_POWER_DBM} dBm via fast-track "
        f"(before={wrong.measured_30dbm:.3f}, after={fixed.measured_30dbm:.3f})"
    )
    # V12/V18: Scaling error must now be within ±25 % (Pass).
    assert abs(fixed.scaling_error) <= 25.0, (
        f"Scaling error must be within ±25 % at correct power {_FAST_TRACK_PASS_POWER_DBM} dBm "
        f"(got {fixed.scaling_error:.1f}%)"
    )
    assert _extract_pssar_result(voila_page.content()) == "Pass", (
        f"psSAR badge must be Pass when power level is corrected to {_FAST_TRACK_PASS_POWER_DBM} dBm"
    )
    body_text = voila_page.locator("body").inner_text()
    assert "Pass rate" in body_text, "Gamma pattern pass-rate section must be present"

    # Restore noise_floor to default so subsequent tests are not affected.
    _set_noise_floor(voila_page, _NOISE_FLOOR_DEFAULT)
    _log("<< test_fast_track_fix_power_restores_pass: pass")
