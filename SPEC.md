# SAR Pattern Validation — Specification

## §G Goal

Recover UI polish + full measurement-validation toolbox from legacy dev branches onto `main-melanie`; regenerate all validation artifacts under current codebase; keep strict py3.9 Voila-frontend / uvx-py3.10+ backend separation throughout.

## §C Constraints

C1: Frontend (`notebooks/voila.ipynb`, `voila_frontend/ui.py`) runs in jupyter-math container → **python 3.9**. No walrus, no match-statement, no 3.10+ stdlib. All UI cherry-picks must pass `ty` type-check with py3.9 target.

C2: Backend (`workflows.py`, `plotting.py`, `report.py`, CLI) runs via **uvx python 3.10+**. Typing and language features at 3.10+ are fine there.

C3: Artifact regeneration runs against current `main-melanie` HEAD (not `develop` / legacy branches). Artifacts must be committed to LFS after regen.

C4: HTML report is pure Python (Jinja2 or stdlib string templates). No `pdflatex` runtime dependency for HTML path; LaTeX/PDF path stays as an optional CLI flag.

C5: LFS scope: measurement CSVs under `data/measurements/` and `data/database/`, artifact `.npz` under `tests/artifacts/`, plot PNGs under `tests/artifacts/measurement_validation/plots/`. All must remain LFS-tracked.

C6: The Voila UI must never re-run registration when only `power_level` changes between consecutive runs (same measured file, same reference, same noise_floor). Power rescaling must be instant (no button cycle). See V6 for the implementation contract.

C7: Adaptive noise floor for measurement validation — when `power_level_dbm ≤ 9`, use `noise_floor = 0.01 W/kg`; otherwise use `0.05 W/kg`. Rationale: at low-to-mid transmit power, SAR amplitudes are small enough that the 0.05 W/kg cutoff excludes too much valid signal and causes `MASK_TOO_SMALL` / `EMPTY_MEASURED_MASK` or inflated gamma failures. Threshold `LOW_POWER_THRESHOLD_DBM = 9`; implemented via `_case_noise_floor()` in `test_measurement_validation.py`.

## §MV Measurement Validation Overview

The measurement validation framework stress-tests the registration + gamma pipeline against real measured SAR data across multiple frequencies, power levels, distances, and averaging masses.

**Data sources**

| Pool | Location | Bands covered |
|------|----------|---------------|
| `BASELINE_CASES` | `data/measurements/D2450_*.csv` (compact format) | 2450 MHz, 10 mm, 1g/10g, 17 dBm, 9 pairs |
| `ROBUSTNESS_CASES` | hard-coded paths in test file | 900 / 1950 / 5800 MHz |
| `DISCOVERED_CASES` | auto-discovered from `data/measurements/` filesystem | all bands present on disk |

**File naming conventions** (two patterns coexist):

- Compact: `D{freq}_Flat_{dist}mm_{power}dBm_{mass}_{index}.csv` (e.g. `D2450_Flat_10mm_17dBm_1g_1.csv`)
- Spacey: `D{freq}_Flat HSL_{dist} mm_{power} dBm_{mass}_{index}.csv` (e.g. `D2450_Flat HSL_10 mm_17 dBm_1g_1.csv`)

`{freq}` may be a plain integer MHz (e.g. `2450`, `900`, `1950`) or `{N}GHz` notation (e.g. `5GHz` → 5800 MHz).

**Artifact layout**

```
tests/artifacts/measurement_validation/
  {frequency_key}/                  # e.g. 2450mhz, 900mhz, 5800mhz
    {power_level_key}/              # e.g. 17dbm, 10dbm
      {case_id}_metrics.json        # scalar metrics dict
      {case_id}.npz                 # gamma_map + evaluation_mask arrays
  plots/
    {case_id}/
      01_loader_comparison.png
      02_registered_measured.png
      02_registration_overlay.png
      03_gamma_map.png
```

Case IDs follow the pattern `{freq_mhz}_{dist}mm_{mass}_{power}dbm_{index}` (e.g. `2450_10mm_1g_17dbm_1`).

**Thresholds** used in `test_measurement_validation.py`:
- `dose_to_agreement = 10 %`, `distance_to_agreement = 3 mm`, `gamma_cap = 2.0`
- `noise_floor`: adaptive via C7 — `0.01 W/kg` when `power_level_dbm ≤ 9`, else `0.05 W/kg`
- **Pass criterion: 100 % gamma pass rate** (`failed_pixel_count == 0`). See V11.

**HTML report** (`scripts/measurement_validation/generate_measurement_validation_report_html.py`): produced from artifact JSON files; filterable by band, power level, pass/fail. Default thresholds: `scaling_error < 10 %`, `gamma_pass_rate = 100 %`.

## §I Interface

I1: `complete_workflow(measured_file_path, reference_file_path, ...)` — runs full registration + gamma pipeline; returns `WorkflowResult` on success, raises `WorkflowExecutionError` with `.issue: ValidationIssue | None` on structured failures.

I2: `ValidationIssue.code` is a machine-readable string (e.g. `MASK_TOO_SMALL`, `EMPTY_MEASURED_MASK`, `CSV_FORMAT_ERROR`). The Voila UI surfaces `issue.message` in the error/warning banner.

I3: `noise_floor` (W/kg) is the SAR floor below which pixels are excluded from the metric mask. Range [0, 0.1]. Must satisfy `noise_floor < measured_peak` for registration to proceed.

I4 (planned): `generate_html_report(results: list[CaseResult], output_path: Path)` — renders filterable HTML table of all measurement-validation cases with pass/fail, gamma pass-rate, and inline thumbnail links.

## §V Invariants

V1: ∀ registration call → `fixed_mask` active pixel count ≥ 1, else raise `ValidationIssue(code="EMPTY_MEASURED_MASK")` before `Execute()`. Applies at `workflows.py` after `make_metric_masks()`.

V2: ∀ `WorkflowExecutionError` raised inside `_complete_workflow` → `.issue` is preserved through exception handlers (no re-wrapping by generic `except Exception` clause). Applied via `except WorkflowExecutionError: raise` as first handler.

V3: `_apply_roi_policy` in `workflows.py` must receive `measured_mask_u8` (SAR ≥ noise cutoff, built by `loader.make_metric_masks()`) as its `measured_mask_u8` arg — never `measured_support_u8` (boundary-only). Gamma eval mask must exclude sub-cutoff (noise-filtered) pixels. Fix: `workflows.py:311`.

V4: ∀ MASK_TOO_SMALL condition (pre-registration on `measured_mask_u8` or post-registration on `evaluator.evaluation_mask`) → raises `WorkflowExecutionError` with `severity="error"` and `code="MASK_TOO_SMALL"`; workflow stops at the first failing check. Pre-registration check fires before `Rigid2DRegistration.run()`.

V5: ∀ cherry-picked frontend commit → must not introduce any python ≥ 3.10 syntax or imports. CI `ty` check with `--python-version 3.9` is the enforcement gate.

V6: When `handle_button_click` detects that only `power_level` changed (same measured-file hash, same reference path, same noise_floor) and a prior `WorkflowResult` exists in memory → skip re-running registration; rescale psSAR via `_update_analytical_results(self.workflow_results)` with the new power level; set banner "Power level updated — results rescaled from prior run." with `severity="info"`. Button must NOT cycle. E2E gate: `test_same_session_rerun_updates_results_after_power_change` detects this by waiting for the unique banner text (not a button cycle).

V7: ∀ artifact regeneration run → artifacts are committed to LFS and the commit message references the `main-melanie` HEAD hash used. Regen must not silently overwrite passing cases with failures without a §B backprop entry.

V8: ∀ E2E CI run → `notebooks/voila.ipynb` must execute in a Jupyter kernel without raising any exception before Playwright tests start. Verified by the `notebook_smoke`-marked pytest step in the `e2e-tests` CI job. Catches syntax errors, ImportErrors, and widget initialisation errors that otherwise surface only as Playwright timeouts.

V9: ∀ E2E Playwright locator targeting a number input by widget identity → must use label-anchored selector (`.widget-text` filtered by `label:has-text(...)`) not positional `nth()`. Positional indices shift whenever a new widget is added to the same DOM row.

V10: ∀ `_compute_case` call in `test_measurement_validation.py` → `noise_floor` must be set adaptively via `_case_noise_floor(case)`: `NOISE_FLOOR_LOW_POWER_WKG = 0.01 W/kg` if `case.power_level_dbm ≤ LOW_POWER_THRESHOLD_DBM = 9`, else `NOISE_FLOOR_WKG = 0.05 W/kg`. Artifact payload records the actual noise floor used.

V11: ∀ measurement validation case → the pass criterion is strictly 100 % gamma pass rate (`failed_pixel_count == 0`). Any failed pixel in the evaluated region is a test failure. Artifacts are not written for failing cases. The HTML report `gamma_pass_rate_threshold_pct` is a display/filter knob and must not be used to soften the per-case assertion.

V12: ∀ fast-track power-level rescale (V6) → "Measured, 30 dBm" and "Scaling Error [%]" must be recomputed from the prior raw measurement at the new power: `new_measured_30dbm = workflow_results.measured_pssar × 10^((old_power_dbm − new_power_dbm)/10)`, `new_scaling_error = (new_measured_30dbm / workflow_results.reference_pssar) − 1`; the psSAR Pass/Fail badge must reflect the recomputed error. E2E gates: `test_fast_track_wrong_power_after_success_shows_failure`, `test_fast_track_fix_power_restores_pass`.

V13: ∀ workflow run where `measurement_area_x_mm` and `measurement_area_y_mm` are set → the measured SAR data must be filtered to `|x_m − cx_m| ≤ measurement_area_x_mm/2000` and `|y_m − cy_m| ≤ measurement_area_y_mm/2000` (where `cx_m`, `cy_m` is the **data midpoint** — `(x_min+x_max)/2`, `(y_min+y_max)/2` of the full unfiltered measured grid, per V19) before mask computation, registration, and gamma evaluation. The plot overlay alone is insufficient — data outside the declared area must not contribute to the gamma result. Applied in `SARImageLoader.__init__`. Unit gate: `test_v13_measurement_area_restricts_data_not_just_plots`.

V14: `measurement_area_x` and `measurement_area_y` Voila widgets must be `widgets.BoundedIntText` (not `widgets.Text`) so they emit `input[type='number']`; `value=0` encodes "auto" (no crop); min=0, max=600/400 enforced by widget. V9 label-anchored selector relies on `input[type='number']`. Gate: all 7 `TestMeasurementAreaInputs` E2E tests.

V15: V12 fast-track E2E tests must reference the baseline passing power via a named constant `_FAST_TRACK_PASS_POWER_DBM`; value must satisfy `|psSAR_scaling_error(measured_sSAR1g.csv, P)| ≤ 25 %` (per V18). Current correct value: 21 dBm (scaling_error ≈ −0.5 %). Gate: `test_fast_track_wrong_power_after_success_shows_failure`, `test_fast_track_fix_power_restores_pass`.

V16: ∀ Compare Patterns click that triggers a full workflow rerun → `result_table.value` must be set to `""` before `update_images(no_data=True)`; the result table must be empty while the run button is disabled, and non-empty once the cycle completes. Gate: `test_result_table_clears_on_rerun_then_repopulates`.

V17: `WorkflowResult` must carry `measured_peak_wkg: float` (= `loader.measured_peak`, noise-filtered max sSAR at measurement power); "Measured, {power} dBm" cell in `_update_analytical_results` must read `workflow_result.measured_peak_wkg` directly — never derive from `measured_pssar` via `×10^((power−30)/10)`. Prevents incorrect display when widget power at display time differs from the power used in the workflow run. Gate: unit test asserting displayed value equals `loader.measured_peak` for a known CSV.

V18: psSAR pass/fail badge (`_update_analytical_results`, notebook cell 11, `pssar_pass` variable) must apply threshold `abs(scaling_error × 100) ≤ 25.0 %` per issue #11 (changed from 10 %). Gate: E2E `test_fast_track_wrong_power_after_success_shows_failure` and `test_fast_track_fix_power_restores_pass` boundary assertions updated to 25.0; `test_fast_track_wrong_power_after_success_shows_failure` must still show Fail at 1 dBm (scaling_error ≈ 9844 %).

V19: when `measurement_area_x_mm` and `measurement_area_y_mm` are specified, the data-filter center and plot window center must be the midpoint of the imported measured grid (`cx_m = (x_max + x_min) / 2`, `cy_m = (y_max + y_min) / 2`) rather than the peak-SAR row. V13 filter criterion amended: `|x_m − cx_m| ≤ x_half_m` where `cx_m` = data midpoint. Prevents half-empty plot window when the peak is near the scan boundary. Locus: `image_loader.py:85-87`.

V20: `noise_floor = 0.0` is a valid input meaning "no noise filtering — all support pixels participate in gamma evaluation"; `WorkflowSchema.noise_floor` must accept `ge=0` (not `gt=0`). `SARImageLoader` already handles zero correctly (`cutoff_wkg = 0`, all-support mask). Gate: schema-level test `test_workflow_schema_accepts_zero_noise_floor`.

V21: overlay legends in Rigid Registration Overlay and Gamma Pass/Fail Map must use `fontsize=7`, `framealpha=0.0`, and label "Noise" (not "Below noise floor") for the noise-floor patch. Reduces overlap with measurement area in tight plots. Locus: `plotting.py:_apply_overlay_legend`, `plotting.py:plot_gamma_results`.

## §T Tasks

Stream A — UI adjustments branch (`jgo/ui-adjustments` from `main-melanie`):

| ID | Status | Task | Cites |
|----|--------|------|-------|
| T1 | x | Create `jgo/ui-adjustments` from `main-melanie` HEAD | |
| T2 | x | Cherry-pick plotting renames + overlays from `develop`: `12cdd09` (Simulated→Reference title), `9746b05` (cropped-area dark-gray overlay + legend), `d24c9d8` (noise-floor medium-gray overlay all 6 panels) | C1,C2,V5 |
| T3 | x | Cherry-pick notebook layout from `develop`: `d774c11` (table below, center plots), `aed839e` (inline banner, swap tables, drop pass/fail button), `264c2d6` (stretch antenna grid to right-column height) | C1,V5 |
| T4 | x | Port boxed log widget + radio-button height limit from `86d7889` (`6.3-noise-floor`); verify py3.9 compat; test scrollable output widget in voila | C1,V5 |
| T5 | x | Run full test suite on `jgo/ui-adjustments`; fix any failures; open PR → `main-melanie` | V5 |

Stream B — Measurement validation toolbox (`main-melanie` direct or sub-branch):

| ID | Status | Task | Cites |
|----|--------|------|-------|
| T7 | x | Recover additional measurement CSVs (1950 / 5800 / 900 MHz bands) + `data/database/` reference CSVs from `develop` or "main"; verify LFS tracking | C5 |
| T8 | x | Extend `test_measurement_validation.py` with recovered bands; add `MeasurementValidationCase` entries for each new dataset | C2,C5 |
| T9 | x | Recover scripts to generate measurement validation HTML report by frequency band and various filtering | C2,C4,I4 |
| T10 | ~ | Regenerate all `tests/artifacts/measurement_validation/` (`.npz` + `_metrics.json` + plot PNGs) under `main-melanie` HEAD with `REGENERATE_MEASUREMENT_VALIDATION_ARTIFACTS=1 SAVE_MEASUREMENT_VALIDATION_PLOTS=1` | C3,C5,V7 |
| T11 | x | Run HTML report over regenerated artifacts; document which cases pass / fail / regress vs `develop` baseline; backprop any new failures via §B | V7,I4 |
| T12 | x | Implement adaptive noise floor in `_compute_case`: `noise_floor = 0.01` when `power_level_dbm ≤ 3`, else `0.05`; re-run T10 for affected cases | C7,V10 |

Stream C — GitHub issue tracker (branch `jgo/m6t4-gamma-excludes-noise-filtered-pixels`):

| ID | Status | Task | Cites |
|----|--------|------|-------|
| T12 | x | #5: remove noise-floor overlay from Reference + gamma panels; `545c487` on this branch | C2,V3 |
| T13 | x | #6: fix Pass legend color in gamma pass/fail map — match actual pass-region white (`plotting.py:518`) | C2 |
| T14 | x | #7: rename axis labels `$x_e$`,`$y_e$` → `$x'_r$`,`$y'_r$` in "Reference, After Registration" and following panels (`plotting.py:77,442,507`; `image_loader.py:548`) | C2 |
| T15 | . | #8: update 1-page PDF report template to revised Overleaf version — colleague task | C4 |
| T16 | x | #9: add `measured_peak_wkg` to `WorkflowResult` (`workflows.py`); populate from `loader.measured_peak`; update `_update_analytical_results` to display it directly as "Measured, {power} dBm" rather than round-tripping through 30 dBm | V17 |
| T17 | x | #11: change psSAR pass/fail threshold from 10 % to 25 % — notebook cell 11 `pssar_pass`; E2E boundary assertions in `test_fast_track_*` | V18 |
| T18 | x | #12: center measurement area window on imported data midpoint rather than peak-SAR location — amend `image_loader.py:85-87` and V13 | V19 |
| T19 | x | #13: allow noise_floor = 0 — change `WorkflowSchema.noise_floor` from `gt=0` to `ge=0`; zero means no noise filtering, all support pixels evaluated | V20 |
| T20 | x | #14: reduce legend overlap — `fontsize=7`, `framealpha=0.0`, rename "Below noise floor" → "Noise" in `_apply_overlay_legend` and `plot_gamma_results` | V21 |

## §M Merge Log

Records every branch merged into `main-melanie`. Critical for squash-merge workflows: a squash-merge rewrites the tip hash, so once a PR is squash-merged the original branch tip listed here is the only reliable way to know what content was included.

| Date | Branch | Tip at merge | What it brought |
|------|--------|-------------|-----------------|
| 2026-05-15 | `jgo/6.6-validation-issue-channel` | `497c17c` | Task 6.6: `ValidationIssue` dataclass + `MASK_TOO_SMALL` / `CSV_FORMAT_ERROR` emit sites; notebook issues-aware banner; banner stdout fix + `status:error` guard; `MASK_TOO_SMALL` E2E test; backprop §B1–§B4, §V1–§V4 |
| 2026-05-15 | `jgo/m6-results-table` | `b84e9a8` | M6 Task 5: two-table results layout in Voila notebook; widget notation fix; CI Voila E2E timeout + dependency updates |
| 2026-05-17 | `jgo/ui-adjustments` | `ee0e493` | T2–T5: plotting renames + overlays (cropped-area + noise-floor), notebook layout (table below, inline banner, stretch grid), boxed log widget + radio-button height limit; power-level fast-path (V6); build backend → hatchling |
| 2026-05-17 | `port/6.2-measurement-area-inputs` | `9606cb5` | Measurement-area X/Y inputs in Voila UI + config validation (gt=22, le=600/400); plot canvas centering; notebook smoke test; label-anchored E2E selectors (V9); §V8–§V9, §B5–§B10 |

Branches already incorporated before this log began (via GitHub PRs, squash-merged onto `main` / `main-melanie`):

| PR | Commit on main-melanie | What it brought |
|----|----------------------|-----------------|
| #18 | `4514399` | Bump actions/checkout 4→6 |
| #17 | `28ede53` | Bump actions/upload-artifact 4→7 |
| #16 | `b44255c` | Update measurement-validation test artifacts after registration direction change |
| #15 | `7b3c702` | User-configurable noise floor input (0 ≤ noise_floor ≤ 0.1 W/kg) |
| #13 | `61c3454` | Task 6.5: inscribed 22×22 mm square mask validity check |
| #8  | `fd5e4c3` | Vectorise gamma; `--output-dir`; lock deps; lint+type CI job; GitLFS for E2E |
| #7  | `ad1595d` | Task 6.4: feedback banners |
| #6  | `ea30322` | Task 6.1: reverse registration direction (gamma in measured frame) |
| #5  | `cf15668` | Scan-for-buttons grid fix; parallel CI stages |
| #4  | `7e33b2a` | Voila E2E Playwright test suite |
| #3  | `69d2bb7` | Run on oSPARC compatibility |
| #1  | `cc77226` | Fix numerical errors in CI |

## §BH Branch Heritage

Full inventory of features in legacy branches relative to `jgo/adding-measurement-validation-back` (current). Use this section to decide what to port.

### `develop` — unique vs current

| Feature | Commit(s) | Port status |
|---------|-----------|-------------|
| Plotting: remove `config.save_colorbars` guard — colorbars always saved | — | not yet |
| Measurement area: 50 mm minimum + blank=auto semantics (string-typed inputs, None when blank) | `703f410` | not yet |
| Measurement area: upper-bound validation (x ≤ 600 mm, y ≤ 400 mm) | `703f410` | not yet |
| RadioButtonGrid `on_filter_changed` callback → reactive run-button enable/disable | — | not yet |
| `_update_run_button()` method (reacts to filter changes without polling) | — | not yet |
| WorkflowResults: removed `min_inscribed_square_mm`, `mask_fits_min_inscribed_square`, `issues` fields | — | not yet |
| `update_images(no_data=True)` call before run starts | — | not yet |
| Noise floor saved to state file on successful run | — | not yet |
| Task 6.9 LaTeX report: `src/sar_pattern_validation/report.py`, `report_template/`, `--report` CLI flag | `fcd71da`–`a1d7e83` | not yet |
| `run_measurement_validation_tests.py` smart rerun script (`--rerun`, `--regenerate-artifacts`, `--save-plots`) | — | not yet |
| `generate_and_open_measurement_validation_dashboard.py` dashboard runner | — | not yet |
| `run_pipeline.py` demo run script | — | not yet |
| `MEASUREMENT_VALIDATION_TESTING.md` test suite docs | — | not yet |
| CI: notebook smoke test removed from e2e job | — | not yet |
| CI: `rm -rf test-artifacts` cleanup after CI run | — | not yet |
| CLI: measurement area help text aligned with 50 mm min | `dab46f2` | not yet |
| root_validator `skip_on_failure=True` | — | not yet |
| Log handler: `clear_logs()` method + stream-output format vs display_data | — | not yet |

### `jgo/feedback-changes` — unique vs current

| Feature | Commit(s) | Port status |
|---------|-----------|-------------|
| Tasks 6.3, 6.6, 6.8, 6.10 in Voila UI (noise floor dropdown, measurement area, PDF download button) | `ff31ec2` | Task 6.3/6.6 via PRs; 6.10 not yet |
| PDF download button (Task 6.10) wired to LaTeX report | — | not yet |
| `.meta.json` loader companion (Task 6.7) — auto-loads measurement parameters from sidecar JSON | `4ed8d23` | not yet |
| Noise floor dropdown with presets (vs continuous float input) | `a621dcd` | not yet |
| `--report_template_dir` CLI flag passed from Voila subprocess | `dd93442` | not yet |
| Additional measurement CSVs: 900 / 1950 / 5800 MHz (from Mark) | `18506d4`–`12c325b` | recovered in T7 |
| `run_measurement_validation_tests.py` (earlier version) | — | superseded by develop |
| `MEASUREMENT_VALIDATION_TESTING.md` | — | superseded by develop version |

### `jgo/feedback-changes-clean` / `remotes/github-jgo/main-jgo-old` — unique vs current

| Feature | Commit(s) | Port status |
|---------|-----------|-------------|
| `INVARIANTS.md` — coupling invariants doc (INV-001 to INV-007) for agents/team | — | not yet |
| `.serena/project.yml` + `.serena/memories/` — Serena onboarding and project memories | — | not yet |
| `MEASUREMENT_VALIDATION_TESTING.md` — test suite documentation | — | superseded by develop |
| Demo run fixture data | `a7edad5` | not yet |
| `kill-voila` Makefile target | `0fa5d50` | not yet |
| Backend log saved to file (`voila_backend.log`) for post-hoc inspection | `6e35806` | not yet |
| Simultaneous handling of local + production Voila paths | `6e35806` | not yet |
| Error hints on frontend | `988a68b` | not yet |
| `ui_state.json` system state file (`notebooks/system_state/`) | — | not yet |
| `.env` file for local dev | — | not yet |
| Avoid rerun for identical data (`_last_run_key` pattern) | `da55fe6` | ported (V6) |

### Other branches — unique vs current

| Branch | Unique content | Notes |
|--------|---------------|-------|
| `exp/geometry-based-init-default` | Geometry-based initialisation as default registration strategy | Research; not stable |
| `codex/type-annotate-tests` | Type annotations for test files | Low priority |
| `port/6.9-report-generation` | LaTeX report (Task 6.9) | Already in `develop` |
| `jgo/m6t4-gamma-excludes-noise-filtered-pixels` | UI adjustments PR #22; merge of github-melanie/main | Already included via T3/T5 |
| `6.3-noise-floor` | User-configurable noise floor input | Already merged as PR #15 |
| `feedback-clean-clean` | `a776bd5` one extra stability commit on top of feedback-changes-clean | Superseded |

### Worktree Inventory

| Path | Branch | Status | Notes |
|------|--------|--------|-------|
| `/home/ordonez/osparc-services/sar-pattern-validation` | `jgo/adding-measurement-validation-back` | active | main worktree |
| `/home/ordonez/osparc-services/sar-pattern-validation-6.3-noise-floor` | `6.3-noise-floor` | prunable | no unique content vs current |
| `/home/ordonez/osparc-services/sar-pattern-validation.worktrees/feedback-clean-clean` | `feedback-clean-clean` | active | superseded by feedback-changes-clean |
| `/home/ordonez/osparc-services/sar-pattern-validation.worktrees/jgo-m6t4-gamma-excludes-noise-filtered-pixels` | `jgo/m6t4-gamma-excludes-noise-filtered-pixels` | active | UI adjustments already ported |
| `.claude/worktrees/agent-a0ac532034dfccd8d` | `worktree-agent-a0ac532034dfccd8d` | locked | stale; all at commit 2e7f706 (feedback-changes-clean era); no unique code |
| `.claude/worktrees/agent-a45f2c9d070ebcbf0` | `worktree-agent-a45f2c9d070ebcbf0` | locked | stale; same as above |
| `.claude/worktrees/agent-a481b6abee89b803e` | `worktree-agent-a481b6abee89b803e` | locked | stale; same as above |
| `.claude/worktrees/agent-a72c75106fb5d802f` | `port/6.9-report-generation` | locked | LaTeX report — content captured in develop |
| `.claude/worktrees/agent-abcf48a3800c04e84` | `worktree-agent-abcf48a3800c04e84` | locked | stale; one commit ahead (705a304) of other agent worktrees |
| `.claude/worktrees/agent-ad1b8857c0dfb0c76` | `worktree-agent-ad1b8857c0dfb0c76` | locked | stale; same as a0ac532 |
| `.claude/worktrees/agent-afaab6b4b3b65e02a` | `worktree-agent-afaab6b4b3b65e02a` | locked | stale; same as a0ac532 |
| `/tmp/sar-gamma-comparison-geometry-based-init` | `exp/geometry-based-init-default` | prunable | research; not stable |
| `/tmp/sar-gamma-comparison-type-tests` | `codex/type-annotate-tests` | prunable | low priority |
| `/tmp/sar-gamma-review-15e1d1f` | detached HEAD `15e1d1f` | prunable | review artifact |

**Cleanup recommendation:** The 6 `worktree-agent-*` locked worktrees under `.claude/worktrees/` and the 3 prunable worktrees in `/tmp/` and `/home/.../sar-pattern-validation-6.3-noise-floor` contain no unique code. Once confirmed safe, `git worktree remove --force` on each. The `feedback-clean-clean` and `jgo-m6t4-*` worktrees should be checked before removal — their branches may have been the source for cherry-picks.

## §T2 Candidate Tasks (from branch diff)

Stream C — Port from `develop`:

| ID | Status | Task | Cites |
|----|--------|------|-------|
| C1 | ✅ done | Port measurement-area 50 mm minimum + blank=auto semantics from `develop:703f410`: change UI inputs to string-typed (blank=None), error banner when < 50 mm, upper bounds (x≤600, y≤400) | |
| C2 | ✅ done | Port `RadioButtonGrid.on_filter_changed` callback + `_update_run_button()` from `develop` — reactive run-button enable/disable without polling; also `radio_button_grid.layout.flex = "0 0 auto"` layout fix | |
| C3 | ⛔ skipped | Remove `config.save_colorbars` guard in `plotting.py` — user decision: keep guard | |
| C4 | ✅ done | Port `update_images(no_data=True)` call before run + save `noise_floor` to state file on success from `develop` | |
| C5 | ⛔ skipped | Port root_validator `skip_on_failure=True` for `FilterOptions.validate_columns` from `develop` — user decision: keep plain `@root_validator` | |
| C6 | ⛔ skipped | Port Task 6.9 LaTeX report: `src/sar_pattern_validation/report.py`, `report_template/`, `--report` CLI flag from `port/6.9-report-generation` (also in `develop`) | C4-SPEC |
| C7 | ✅ done | Port `run_measurement_validation_tests.py` smart rerun script (cherry-picked from `jgo/feedback-changes-clean`) | |
| C8 | ✅ done | Port `generate_and_open_measurement_validation_dashboard.py` and `MEASUREMENT_VALIDATION_TESTING.md` (cherry-picked from `jgo/feedback-changes-clean`) | |
| C9 | ✅ done | CI: remove `notebook_smoke` job from e2e workflow; playwright output → `tests/artifacts/playwright/`; add `.gitignore` entry | |
| C10 | ✅ done | Port log handler `clear_logs()` method + stream-output format (vs display_data) from `develop` | |

Stream D — Port from `jgo/feedback-changes-clean`:

| ID | Status | Task | Cites |
|----|--------|------|-------|
| D1 | ⛔ skipped | Add `INVARIANTS.md` from `jgo/feedback-changes-clean` — user decision: check SPEC instead | |
| D2 | ⛔ skipped | Add `.serena/project.yml` + `.serena/memories/` from `jgo/feedback-changes-clean` for Serena onboarding | |
| D3 | ✅ done | Port `kill-voila` Makefile target — already present in `Makefile` from commit `ab7ad56` | |
| D4 | ✅ done | Port backend log saved to file (`voila_backend.log`) from `jgo/feedback-changes-clean:6e35806` | |

Stream E — Port from `jgo/feedback-changes`:

| ID | Status | Task | Cites |
|----|--------|------|-------|
| E1 | ⛔ skipped | Port PDF download button (Task 6.10) from `jgo/feedback-changes` — user decision: skip | C6 |
| E2 | ⛔ skipped | Port `.meta.json` companion loader (Task 6.7) from `jgo/feedback-changes:4ed8d23` — user decision: skip | |

## §B Bug Log

| ID | Date | Root cause | Invariant |
|----|------|-----------|-----------|
| B12 | 2026-05-18 | `measurement_area_x_mm`/`measurement_area_y_mm` passed to `WorkflowConfig` but never forwarded to `SARImageLoader`; full CSV always used for registration and gamma. Plot overlay marks outside data as "Cropped" but gamma silently uses it, producing a spurious 100 % pass even when the declared area contains no SAR above noise floor | V13 |
| B11 | 2026-05-18 | Fast-track path (V6) passes stale `workflow_results` (computed at old power level) directly to `_update_analytical_results`; only `measured_at_power` (first column) recalculates using the fresh `power_level.value`; "Measured, 30 dBm" and "Scaling Error [%]" stay frozen at prior-run values, so the psSAR Pass/Fail badge reflects the wrong power | V12 |
| B1 | 2026-05-15 | `noise_floor ≥ measured peak` → empty fixed mask → `VirtualSampledPointSet must have 1 or more points` crash in SimpleITK, surfaced as raw ITK traceback in Voila banner | V1 |
| B2 | 2026-05-15 | `_complete_workflow` generic `except Exception` handler re-wrapped `WorkflowExecutionError` raised from inside the `try` block, discarding `.issue` | V2 |
| B3 | 2026-05-15 | `workflows.py:311` passes `measured_support_u8` (boundary-only) to `_apply_roi_policy` instead of `measured_mask_u8`; noise-filtered (SAR < cutoff) pixels included in gamma eval mask → inflated pass rate (Task 6.4) | V3 |
| B4 | 2026-05-15 | MASK_TOO_SMALL checked only post-registration; pre-registration noise-filtered `measured_mask_u8` never verified against `min_inscribed_square_mm` | V4 |
| B5 | 2026-05-15 | MASK_TOO_SMALL emitted as `severity="warning"` appended to `issues`, allowing workflow to complete; physically it is a hard validity gate — comparison on a sub-22mm mask is invalid | V4 |
| B6 | 2026-05-15 | `widgets.Layout(align_items="flex_start")` — underscore instead of CSS hyphen — caused voila to fail at startup; all E2E Playwright tests timed out rather than showing a useful error | V8 |
| B7 | 2026-05-16 | `84ae861` merge on `jgo/m6-results-table` silently dropped 5 noise_floor lines: method def (→ AttributeError), run-key entry (→ stale cache on floor change), `restore_state` read+set (→ lost on reload), `top_row` flex_item (→ widget invisible in UI) | V8 |
| B8 | 2026-05-16 | `_set_meas_area` and two upper-bound tests used `input[type='number'].nth(1/2)`; adding `noise_floor` widget to `top_row` (§B7 fix) shifted DOM order → inputs resolved to wrong widget, values clamped to 0.1 max, tests timed out | V9 |
| B9 | 2026-05-16 | `test_workflow_produces_square_plots` unpacked `voila_server` as 2-tuple (`_, workspace_root`) but fixture yields 3-tuple; raised `ValueError: too many values to unpack` | — |
| B10 | 2026-05-16 | `84ae861` merge dropped `measurement_area_row` from `left_setup_section` in `create_ui`; `measurement_area_x/y` widgets were defined but never added to the DOM → Playwright locators timed out finding them | V9 |
| B13 | 2026-05-18 | `measurement_area_x/y` widgets created as `widgets.Text` (renders `input[type='text']`); `_meas_area_input` selector targets `input[type='number']` → all 7 `TestMeasurementAreaInputs` timeout; auto-detect logic used empty-string check incompatible with `BoundedIntText.value=0` | V14 |
| B14 | 2026-05-18 | Fast-track E2E tests hardcoded `23.0` dBm as "correct power" for `measured_sSAR1g.csv`; at 23 dBm `scaling_error = −37.3 %` → assertion `\|scaling_error\| ≤ 10 %` fails; correct power is 21 dBm (`scaling_error ≈ −0.5 %`; `raw_peak ≈ 5.23 W/kg × 10^(9/10) ≈ 41.5 W/kg ≈ reference 41.76 W/kg`) | V15 |
| B15 | 2026-05-18 | On a second Compare Patterns click `result_table.value` was not cleared; stale result table stayed visible while images were blanked, giving a misleading mixed state during the run | V16 |
| B16 | 2026-05-19 | "Measured, {power} dBm" psSAR cell derived via `measured_pssar × 10^((run_power − 30)/10)` (round-trip through 30 dBm) instead of storing the at-power peak; `WorkflowResult` has no `measured_peak_wkg` field, so the current widget power (which may differ from run power in fast-track) silently corrupts the displayed value | V17 |
| B17 | 2026-05-19 | measurement area window centered on peak-SAR location (`image_loader.py:86`, per V13) rather than the midpoint of the imported measured grid; when the measurement scan is asymmetric (peak near boundary), up to half the plot window shows empty space outside the actual scan range | V19 |
| B18 | 2026-05-19 | `WorkflowSchema.noise_floor` declared `gt=0` (strictly positive) but the widget allows `min=0.0`; entering 0 and clicking Compare Patterns raises Pydantic `ValidationError` shown as a raw error banner — should silently mean "no noise filtering" | V20 |
| B19 | 2026-05-19 | overlay legend in Rigid Registration Overlay and Gamma Pass/Fail Map uses `fontsize=9` and opaque frame (`framealpha` default ≈ 0.8), occupying too much space and overlapping the measurement area; label "Below noise floor" is verbose | V21 |
