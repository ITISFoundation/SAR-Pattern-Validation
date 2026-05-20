#!/usr/bin/env python
"""Generate a measurement-validation HTML dashboard from one or more JSON reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

DEFAULT_SCALING_ERROR_THRESHOLD_PCT: Final[float] = 25.0
DEFAULT_GAMMA_PASS_RATE_THRESHOLD_PCT: Final[float] = 100.0


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _status_class(status: str) -> str:
    if status == "passed":
        return "passed"
    if status == "failed":
        return "failed"
    return "error"


def _combined_verdict(
    case: dict[str, Any],
    *,
    scaling_threshold_pct: float,
    gamma_threshold_pct: float,
) -> str:
    """Compute combined pass/fail per Task 6.8.

    Both gamma pass-rate and scaling-error criteria must pass. When the case is
    in error state or required metrics are missing, returns "error".
    """
    if str(case.get("status")) == "error":
        return "error"
    pass_rate = case.get("pass_rate_percent")
    scaling_error = case.get("scaling_error")
    if pass_rate is None or scaling_error is None:
        return "error"
    try:
        gamma_pass = float(pass_rate) >= gamma_threshold_pct
        scaling_pass = abs(float(scaling_error) * 100.0) <= scaling_threshold_pct
    except (TypeError, ValueError):
        return "error"
    return "passed" if (gamma_pass and scaling_pass) else "failed"


def _pass_rate_class(pass_rate: float | None) -> str:
    if pass_rate is None:
        return "unknown"
    if pass_rate >= 99:
        return "good"
    if pass_rate >= 90:
        return "partial"
    return "bad"


def _format_validation_issues_html(case: dict[str, Any]) -> str:
    issues = case.get("validation_issues") or []
    if not isinstance(issues, list) or not issues:
        return ""
    parts: list[str] = ["<ul class='issues'>"]
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "error").lower()
        if severity not in ("error", "warning", "info"):
            severity = "error"
        code = _escape_html(str(issue.get("code") or "")).strip()
        message = _escape_html(str(issue.get("message") or "")).strip()
        label = code or severity.upper()
        parts.append(
            f"<li class='issue {severity}'><span class='issue-code'>{label}</span> {message}</li>"
        )
    parts.append("</ul>")
    return "".join(parts)


def _format_dbm_label(value: Any) -> str:
    try:
        numeric = float(value)
        if numeric.is_integer():
            return f"{int(numeric)} dBm"
        return f"{numeric:g} dBm"
    except (TypeError, ValueError):
        return f"{value} dBm"


def _resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.input or []:
        path = Path(raw)
        if path.exists() and path.is_file() and path.suffix == ".json":
            paths.append(path)

    if args.input_glob:
        for path in sorted(Path().glob(args.input_glob)):
            if path.is_file() and path.suffix == ".json":
                paths.append(path)

    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)

    return unique


def _load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        reports.append(
            {
                "path": path,
                "name": path.name,
                "generated_at": data.get("generated_at", ""),
                "report": data,
            }
        )
    reports.sort(key=lambda entry: (entry["generated_at"], entry["name"]))
    return reports


def _aggregate_cases(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case_id: dict[str, dict[str, Any]] = {}
    for source in reports:
        for case in source["report"].get("cases", []):
            enriched = dict(case)
            enriched["_report_name"] = source["name"]
            by_case_id[str(case["case_id"])] = enriched

    return sorted(
        by_case_id.values(),
        key=lambda case: (
            int(case.get("frequency_mhz", 0)),
            float(case.get("power_level_dbm", 0.0)),
            int(case.get("distance_mm", 0)),
            str(case.get("averaging_mass", "")),
            str(case.get("case_id", "")),
        ),
    )


def _build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    passed = sum(1 for c in cases if c.get("status") == "passed")
    failed = sum(1 for c in cases if c.get("status") == "failed")
    error = sum(1 for c in cases if c.get("status") == "error")
    total_evaluated = sum(int(c.get("evaluated_pixel_count", 0)) for c in cases)
    total_failed_pixels = sum(int(c.get("failed_pixel_count", 0)) for c in cases)
    pass_rate = (
        100.0 * (total_evaluated - total_failed_pixels) / total_evaluated
        if total_evaluated
        else None
    )
    case_pass_rate = 100.0 * passed / total if total else None

    return {
        "case_count": total,
        "passed_case_count": passed,
        "failed_case_count": failed,
        "error_case_count": error,
        "case_pass_rate_percent": case_pass_rate,
        "aggregate_pass_rate_percent": pass_rate,
    }


def _group_cases_by_frequency(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        key = str(case.get("frequency_key", "unknown"))
        if key == "unknown":
            continue
        grouped[key].append(case)

    frequencies: list[dict[str, Any]] = []
    for _, frequency_cases in sorted(
        grouped.items(), key=lambda e: int(e[1][0].get("frequency_mhz", 0))
    ):
        exemplar = frequency_cases[0]
        frequencies.append(
            {
                "frequency_key": exemplar.get("frequency_key", "unknown"),
                "frequency_label": exemplar.get("frequency_label", "Unknown"),
                "frequency_mhz": exemplar.get("frequency_mhz", 0),
                "summary": _build_summary(frequency_cases),
                "cases": frequency_cases,
            }
        )

    return frequencies


def _generate_html(
    reports: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    scaling_threshold_pct: float = DEFAULT_SCALING_ERROR_THRESHOLD_PCT,
    gamma_threshold_pct: float = DEFAULT_GAMMA_PASS_RATE_THRESHOLD_PCT,
) -> str:
    summary = _build_summary(cases)
    frequencies = _group_cases_by_frequency(cases)

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='en'>")
    parts.append("<head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>Measurement Validation Dashboard</title>")
    parts.append("<style>")
    parts.append(
        """
        :root {
            --bg: #0f172a;
            --panel: #f8fafc;
            --ink: #0b1020;
            --brand: #0ea5e9;
            --ok: #15803d;
            --warn: #b45309;
            --bad: #b91c1c;
        }
        body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background: radial-gradient(circle at top, #1e293b, var(--bg)); color: var(--ink); }
        .wrap { max-width: 1400px; margin: 0 auto; padding: 24px; }
        .panel { background: var(--panel); border-radius: 14px; padding: 20px; box-shadow: 0 12px 40px rgba(0,0,0,.25); }
        h1 { margin: 0 0 8px 0; }
        .muted { color: #4b5563; font-size: 13px; }
        .cards { margin-top: 16px; display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
        .card { background: #e2e8f0; border-radius: 10px; padding: 12px; }
        .card .label { font-size: 12px; text-transform: uppercase; color: #334155; }
        .card .value { font-size: 28px; font-weight: 700; }
        .ok { color: var(--ok); }
        .bad { color: var(--bad); }
        .warn { color: var(--warn); }
        .controls { margin: 18px 0; display: grid; gap: 10px; }
        .chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
        .chip { border: 1px solid #cbd5e1; background: white; padding: 6px 10px; border-radius: 999px; cursor: pointer; font-size: 13px; }
        .chip.active { background: #bae6fd; border-color: #7dd3fc; }
        .status-row label { margin-right: 14px; font-size: 14px; }
        .section { margin-top: 18px; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }
        .section h2 { margin: 0; padding: 12px 14px; background: #f1f5f9; font-size: 18px; display: flex; justify-content: space-between; align-items: center; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px 10px; border-top: 1px solid #e5e7eb; font-size: 13px; text-align: left; }
        th { background: #f8fafc; position: sticky; top: 0; }
        .badge { border-radius: 999px; padding: 2px 8px; font-weight: 700; font-size: 11px; text-transform: uppercase; }
        .badge.passed { background: #dcfce7; color: #166534; }
        .badge.failed { background: #fee2e2; color: #991b1b; }
        .badge.error { background: #fef3c7; color: #92400e; }
        .small { font-size: 12px; color: #334155; }
        .hide { display: none; }
        ul.issues { list-style: none; margin: 4px 0 0 0; padding: 0; }
        ul.issues li.issue { font-size: 12px; padding: 4px 8px; border-radius: 6px; margin-top: 4px; border-left: 3px solid #cbd5e1; }
        ul.issues li.issue.error { background: #fef2f2; border-left-color: #b91c1c; color: #7f1d1d; }
        ul.issues li.issue.warning { background: #fffbeb; border-left-color: #b45309; color: #78350f; }
        ul.issues li.issue.info { background: #eff6ff; border-left-color: #2563eb; color: #1e3a8a; }
        ul.issues .issue-code { font-weight: 700; margin-right: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        @media (max-width: 900px) { .small-col { display: none; } }
        """
    )
    parts.append("</style>")
    parts.append("</head>")
    parts.append("<body><div class='wrap'><div class='panel'>")
    parts.append("<h1>Measurement Validation Dashboard</h1>")
    parts.append(
        f"<div class='muted'>Sources: {len(reports)} JSON report(s). Cases shown: {summary['case_count']}.</div>"
    )

    pass_rate = summary["case_pass_rate_percent"]
    pass_rate_text = f"{pass_rate:.2f}%" if pass_rate is not None else "N/A"
    pass_rate_class = _pass_rate_class(pass_rate)
    combined_pass_count = sum(
        1
        for case in cases
        if _combined_verdict(
            case,
            scaling_threshold_pct=scaling_threshold_pct,
            gamma_threshold_pct=gamma_threshold_pct,
        )
        == "passed"
    )
    parts.append("<div class='cards'>")
    parts.append(
        f"<div class='card'><div class='label'>Total Cases</div><div class='value'>{summary['case_count']}</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>Passed</div><div class='value ok'>{summary['passed_case_count']}</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>Failed</div><div class='value bad'>{summary['failed_case_count']}</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>Errors</div><div class='value warn'>{summary['error_case_count']}</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>Aggregate Pass Rate</div><div class='value {pass_rate_class}'>{pass_rate_text}</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>Combined Pass</div><div class='value ok'>{combined_pass_count}</div>"
        f"<div class='small'>scaling ≤ ±{scaling_threshold_pct:g}% &amp; gamma ≥ {gamma_threshold_pct:g}%</div></div>"
    )
    parts.append("</div>")

    parts.append("<div class='controls'>")
    parts.append("<div><strong>Frequency Toggles</strong></div>")
    parts.append("<div class='chip-row'>")
    parts.append("<button class='chip active' data-frequency='all'>All</button>")
    for freq in frequencies:
        key = _escape_html(str(freq["frequency_key"]))
        label = _escape_html(str(freq["frequency_label"]))
        parts.append(f"<button class='chip' data-frequency='{key}'>{label}</button>")
    parts.append("</div>")

    parts.append("<div><strong>dBm Toggles</strong></div>")
    parts.append("<div class='chip-row' id='dbm-chip-row'>")
    parts.append("<button class='chip active' data-dbm='all'>All</button>")
    parts.append("</div>")

    parts.append("<div class='status-row'><strong>Status Filters</strong><br>")
    parts.append(
        "<label><input type='checkbox' class='status-filter' value='passed' checked> passed</label>"
    )
    parts.append(
        "<label><input type='checkbox' class='status-filter' value='failed' checked> failed</label>"
    )
    parts.append(
        "<label><input type='checkbox' class='status-filter' value='error' checked> error</label>"
    )
    parts.append("</div></div>")

    for freq in frequencies:
        freq_key = _escape_html(str(freq["frequency_key"]))
        freq_label = _escape_html(str(freq["frequency_label"]))
        freq_summary = freq["summary"]
        freq_pass_rate = freq_summary["case_pass_rate_percent"]
        freq_pass_rate_text = (
            f"{freq_pass_rate:.2f}%" if freq_pass_rate is not None else "N/A"
        )

        parts.append(f"<section class='section' data-frequency-section='{freq_key}'>")
        parts.append(
            "<h2>"
            f"<span>{freq_label}</span>"
            f"<span class='small'>pass {freq_summary['passed_case_count']}/{freq_summary['case_count']} | fail {freq_summary['failed_case_count']} | rate {freq_pass_rate_text}</span>"
            "</h2>"
        )
        parts.append("<div style='overflow:auto; max-height: 62vh;'>")
        parts.append("<table><thead><tr>")
        parts.append(
            "<th>Case ID</th><th>dBm</th><th>Distance</th><th>Mass</th><th>Noise Floor</th><th class='small-col'>Evaluated</th><th>Failed Pixels</th><th>Pass Rate</th><th>Scaling Err [%]</th><th class='small-col'>Source JSON</th>"
        )
        parts.append("</tr></thead><tbody>")

        for case in freq["cases"]:
            status = str(case.get("status", "error"))
            status_class = _status_class(status)
            case_pass_rate = case.get("pass_rate_percent")
            case_pass_rate_text = (
                f"{float(case_pass_rate):.2f}%" if case_pass_rate is not None else "N/A"
            )
            failed_pixels = int(case.get("failed_pixel_count", 0))
            evaluated = int(case.get("evaluated_pixel_count", 0))

            combined = _combined_verdict(
                case,
                scaling_threshold_pct=scaling_threshold_pct,
                gamma_threshold_pct=gamma_threshold_pct,
            )
            combined_class = _status_class(combined)
            scaling_error = case.get("scaling_error")
            scaling_error_text = (
                f"{float(scaling_error) * 100.0:.2f}"
                if scaling_error is not None
                else "N/A"
            )

            noise_floor = case.get("noise_floor_wkg")
            noise_floor_text = (
                f"{noise_floor} W/kg" if noise_floor is not None else "N/A"
            )

            issues_html = _format_validation_issues_html(case)
            case_id_cell = f"<code>{_escape_html(str(case.get('case_id', '')))}</code>"
            if issues_html:
                case_id_cell = f"{case_id_cell}{issues_html}"
            parts.append(
                "<tr "
                f"data-frequency='{freq_key}' "
                f"data-dbm='{_escape_html(_format_dbm_label(case.get('power_level_dbm', 'N/A')))}' "
                f"data-status='{_escape_html(status)}' "
                f"data-combined='{_escape_html(combined)}'>"
                f"<td>{case_id_cell}</td>"
                f"<td>{_escape_html(_format_dbm_label(case.get('power_level_dbm', 'N/A')))}</td>"
                f"<td>{_escape_html(str(case.get('distance_mm', 'N/A')))} mm</td>"
                f"<td>{_escape_html(str(case.get('averaging_mass', 'N/A')))}</td>"
                f"<td>{noise_floor_text}</td>"
                f"<td class='small-col'>{evaluated:,}</td>"
                f"<td>{failed_pixels:,}</td>"
                f"<td><span class='{_pass_rate_class(case_pass_rate)}'>{case_pass_rate_text}</span> <span class='badge {status_class}'>{_escape_html(status)}</span></td>"
                f"<td>{scaling_error_text} <span class='badge {combined_class}'>{_escape_html(combined)}</span></td>"
                f"<td class='small-col'>{_escape_html(str(case.get('_report_name', '')))}</td>"
                "</tr>"
            )

        parts.append("</tbody></table></div></section>")

    parts.append("<script>")
    parts.append(
        """
        const frequencyChips = Array.from(document.querySelectorAll('[data-frequency]'));
        const dbmChipRow = document.getElementById('dbm-chip-row');
        const statusChecks = Array.from(document.querySelectorAll('.status-filter'));
        const rows = Array.from(document.querySelectorAll('tbody tr'));
        const sections = Array.from(document.querySelectorAll('[data-frequency-section]'));

        let activeFrequency = 'all';
        let activeDbm = 'all';

        function availableDbmForActiveFrequency() {
            const set = new Set();
            rows.forEach(row => {
                const matchesFrequency = activeFrequency === 'all' || row.dataset.frequency === activeFrequency;
                if (matchesFrequency) {
                    set.add(row.dataset.dbm);
                }
            });
            return Array.from(set).sort((a, b) => parseFloat(a) - parseFloat(b));
        }

        function rebuildDbmChips() {
            const dbmOptions = availableDbmForActiveFrequency();
            dbmChipRow.innerHTML = '';

            const allBtn = document.createElement('button');
            allBtn.className = 'chip' + (activeDbm === 'all' ? ' active' : '');
            allBtn.dataset.dbm = 'all';
            allBtn.textContent = 'All';
            allBtn.addEventListener('click', () => {
                activeDbm = 'all';
                rebuildDbmChips();
                applyFilters();
            });
            dbmChipRow.appendChild(allBtn);

            dbmOptions.forEach(dbm => {
                const btn = document.createElement('button');
                btn.className = 'chip' + (activeDbm === dbm ? ' active' : '');
                btn.dataset.dbm = dbm;
                btn.textContent = dbm;
                btn.addEventListener('click', () => {
                    activeDbm = dbm;
                    rebuildDbmChips();
                    applyFilters();
                });
                dbmChipRow.appendChild(btn);
            });

            if (activeDbm !== 'all' && !dbmOptions.includes(activeDbm)) {
                activeDbm = 'all';
                rebuildDbmChips();
            }
        }

        function selectedStatuses() {
            return new Set(statusChecks.filter(c => c.checked).map(c => c.value));
        }

        function updateFrequencyChipStyles() {
            frequencyChips.forEach(chip => {
                chip.classList.toggle('active', chip.dataset.frequency === activeFrequency);
            });
        }

        function applyFilters() {
            const statuses = selectedStatuses();
            rows.forEach(row => {
                const freq = row.dataset.frequency;
                const dbm = row.dataset.dbm;
                const status = row.dataset.status;
                const freqVisible = activeFrequency === 'all' || activeFrequency === freq;
                const dbmVisible = activeDbm === 'all' || activeDbm === dbm;
                const statusVisible = statuses.has(status);
                row.classList.toggle('hide', !(freqVisible && dbmVisible && statusVisible));
            });

            sections.forEach(section => {
                const freq = section.dataset.frequencySection;
                const visibleRows = section.querySelectorAll('tbody tr:not(.hide)').length;
                const sectionVisible = (activeFrequency === 'all' || activeFrequency === freq) && visibleRows > 0;
                section.classList.toggle('hide', !sectionVisible);
            });
        }

        frequencyChips.forEach(chip => {
            chip.addEventListener('click', () => {
                activeFrequency = chip.dataset.frequency;
                activeDbm = 'all';
                updateFrequencyChipStyles();
                rebuildDbmChips();
                applyFilters();
            });
        });

        statusChecks.forEach(check => check.addEventListener('change', applyFilters));

        updateFrequencyChipStyles();
        rebuildDbmChips();
        applyFilters();
        """
    )
    parts.append("</script>")
    parts.append("</div></div></body></html>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an HTML dashboard from one or more measurement-validation JSON reports."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Path to a JSON report file. Can be repeated.",
    )
    parser.add_argument(
        "--input-glob",
        help="Glob for JSON report files, e.g. tests/artifacts/measurement_validation/reports/*mhz.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output HTML file.",
    )
    parser.add_argument(
        "--scaling-error-threshold-pct",
        type=float,
        default=DEFAULT_SCALING_ERROR_THRESHOLD_PCT,
        help=(
            "Scaling-error pass threshold in percent (default 10). "
            "Combined verdict passes only when |scaling_error_pct| is at or below this value."
        ),
    )
    parser.add_argument(
        "--gamma-pass-rate-threshold-pct",
        type=float,
        default=DEFAULT_GAMMA_PASS_RATE_THRESHOLD_PCT,
        help="Gamma pass-rate threshold in percent (default 100).",
    )
    args = parser.parse_args(argv)

    input_paths = _resolve_input_paths(args)
    if not input_paths:
        print("Error: no input JSON files found.", file=sys.stderr)
        return 1

    try:
        reports = _load_reports(input_paths)
    except json.JSONDecodeError as error:
        print(f"Error: failed to parse JSON: {error}", file=sys.stderr)
        return 1

    cases = _aggregate_cases(reports)
    html = _generate_html(
        reports,
        cases,
        scaling_threshold_pct=args.scaling_error_threshold_pct,
        gamma_threshold_pct=args.gamma_pass_rate_threshold_pct,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"HTML dashboard generated: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
