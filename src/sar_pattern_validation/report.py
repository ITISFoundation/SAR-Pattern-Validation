"""
Generate SAR Pattern Validation reports from workflow results.

The report uses tested_case_report_page_new/main.tex as a standalone template.
On the first workflow run the template main.tex is copied into the output
directory.  Subsequent runs append additional test-case pages before the
\\end{document} marker in the same output directory.

Output path is exposed for [[Task 6.10 - User Report Download Button]] (MEST).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess  # nosec B404
from pathlib import Path

from sar_pattern_validation.workflow_config import WorkflowConfig
from sar_pattern_validation.workflows import WorkflowResult

LOGGER = logging.getLogger(__name__)

DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "report_template"

# Plot-file mapping: template figure name -> WorkflowResult attribute holding
# the source path. Order matters only for documentation; lookups are by key.
TEMPLATE_FIGURE_MAPPING: dict[str, str] = {
    "gamma_index_with_colorbar.png": "gamma_image_path",
    "gamma_failures.png": "failure_image_path",
    "registration_nocolorbar.png": "registered_overlay_path",
    "measured_with_colorbar.png": "measured_image_path",
    "reference_with_colorbar.png": "reference_image_path",
}

# Marker in main.tex where test-case content is inserted.
_DOCUMENT_END_MARKER = r"\end{document}"


def _auto_measurement_area_mm(measured_csv_path: str | Path) -> tuple[float, float]:
    """
    Return (x_extent_mm, y_extent_mm) of the measured CSV's coordinate grid.

    Used as the report fallback when no explicit measurement area was supplied:
    the workflow processes the full extent of the measured grid in that case,
    so we mirror that here. ``SARImageLoader._read_csv`` normalises x/y to
    meters regardless of the original header units.
    """
    # Local import to avoid pulling SimpleITK at module import time for callers
    # that only need the public report API surface.
    from sar_pattern_validation.image_loader import SARImageLoader

    df = SARImageLoader._read_csv(str(measured_csv_path))
    x_mm = float((df["x_m"].max() - df["x_m"].min()) * 1000.0)
    y_mm = float((df["y_m"].max() - df["y_m"].min()) * 1000.0)
    return x_mm, y_mm


def _set_latex_macro(text: str, name: str, value: str) -> str:
    """
    Substitute the body of a LaTeX macro definition. Handles both
    `\\newcommand{\\NAME}{...}` and `\\def\\NAME{...}` (the template uses
    `\\def` for ``\\passrate`` because of the FPeval branching below it).
    """
    newcmd = re.compile(r"\\newcommand\{\\" + re.escape(name) + r"\}\{[^}]*\}")
    if newcmd.search(text):
        return newcmd.sub(lambda _m: f"\\newcommand{{\\{name}}}{{{value}}}", text)
    def_re = re.compile(r"\\def\\" + re.escape(name) + r"\{[^}]*\}")
    return def_re.sub(lambda _m: f"\\def\\{name}{{{value}}}", text)


def _latex_escape_filename(name: str) -> str:
    """Escape characters that LaTeX interprets specially in a typewritten filename."""
    return (
        name.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def compile_report(main_tex: Path) -> Path | None:
    """
    Compile *main_tex* to PDF with pdflatex (+ biber if biblatex is used).

    Runs: pdflatex → biber (if needed) → pdflatex → pdflatex
    This ensures bibliography references and cross-references are resolved.

    pdflatex must be on PATH; if it isn't, logs a warning and returns None so
    the caller can fall back to distributing the .tex file instead.

    Returns the path to the compiled PDF, or None if compilation is skipped or
    fails.
    """
    if not shutil.which("pdflatex"):
        LOGGER.warning(
            "pdflatex not found on PATH — skipping PDF compilation; "
            "distribute %s instead",
            main_tex,
        )
        return None

    output_dir = main_tex.parent
    pdflatex_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        main_tex.name,
    ]

    # First pdflatex pass (generates .aux/.bcf for biber, or .aux for bibtex)
    if not _run_latex_cmd(pdflatex_cmd, output_dir, "pdflatex pass 1"):
        return None

    # Run biber if the document uses biblatex (produces a .bcf file),
    # otherwise run bibtex for traditional bibliography.
    bcf_file = output_dir / main_tex.with_suffix(".bcf").name
    if bcf_file.is_file() and shutil.which("biber"):
        biber_cmd = ["biber", main_tex.stem]
        if not _run_latex_cmd(biber_cmd, output_dir, "biber"):
            LOGGER.warning("biber failed; PDF will lack bibliography")
    elif shutil.which("bibtex"):
        bibtex_cmd = ["bibtex", main_tex.stem]
        if not _run_latex_cmd(bibtex_cmd, output_dir, "bibtex"):
            LOGGER.warning("bibtex failed; PDF will lack bibliography")

    # Second and third pdflatex passes (resolve references)
    for pass_num in (2, 3):
        if not _run_latex_cmd(pdflatex_cmd, output_dir, f"pdflatex pass {pass_num}"):
            return None

    pdf_path = output_dir / main_tex.with_suffix(".pdf").name
    if not pdf_path.is_file():
        LOGGER.error("pdflatex exited 0 but %s not found", pdf_path)
        return None

    LOGGER.info("PDF compiled: %s", pdf_path)
    return pdf_path


def _run_latex_cmd(cmd: list[str], cwd: Path, label: str) -> bool:
    """Run a LaTeX toolchain command; return True on success."""
    try:
        result = subprocess.run(  # nosec B603
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        LOGGER.error("%s timed out after 120 s", label)
        return False

    if result.returncode != 0:
        LOGGER.error(
            "%s failed (rc=%d):\n%s",
            label,
            result.returncode,
            result.stdout[-2000:],
        )
        return False
    return True


def _initialize_report(output_dir: Path, template_dir: Path) -> None:
    """
    Copy the report template into *output_dir* for the first workflow run.

    Only the package imports are kept; the sample macro definitions and document
    body are stripped because the Python-rendered test cases provide all content
    with values resolved inline.
    """
    template_file = template_dir / "main.tex"
    if not template_file.is_file():
        raise FileNotFoundError(f"Report template not found: {template_file}")

    template_text = template_file.read_text(encoding="utf-8")

    # Extract only the package/setup lines (up to "% input variables" comment
    # or to \begin{document}), then write a minimal document shell.
    lines = template_text.splitlines(keepends=True)
    preamble_lines: list[str] = []
    for line in lines:
        # Stop before sample variable definitions
        if line.strip().startswith("% input variables"):
            break
        preamble_lines.append(line)

    preamble = "".join(preamble_lines)
    report_text = preamble + "\n\\begin{document}\n\\end{document}\n"
    (output_dir / "main.tex").write_text(report_text, encoding="utf-8")


def _next_case_number(output_dir: Path) -> int:
    """Determine the next case number by counting existing case_* figure dirs."""
    figures_dir = output_dir / "figures"
    if not figures_dir.is_dir():
        return 0
    existing = sorted(
        d.name
        for d in figures_dir.iterdir()
        if d.is_dir() and d.name.startswith("case_")
    )
    return len(existing)


def _render_test_case_body(
    *,
    workflow_result: WorkflowResult,
    workflow_config: WorkflowConfig,
    antenna_type: str,
    frequency_mhz: int,
    distance_mm: int,
    mass_g: int,
    figures_relpath: str,
) -> str:
    """
    Generate the LaTeX content for a single tested-case subsection.

    The content is derived from the tested_case_report_page template with all
    macro values resolved inline (no \\newcommand definitions needed).
    """
    measured_filename = _latex_escape_filename(
        Path(workflow_config.measured_file_path).name
    )
    noise_level = f"{workflow_config.noise_floor:g}"
    # Resolve the measurement area written to the report. If the GUI/CLI
    # supplied explicit dimensions, use them. Otherwise fall back to the
    # auto-detected extent of the measured CSV (max-min of x/y, in mm) so
    # the report reflects the actual area the workflow processed instead of
    # showing dashes.
    area_x_mm = workflow_config.measurement_area_x_mm
    area_y_mm = workflow_config.measurement_area_y_mm
    if area_x_mm is None or area_y_mm is None:
        try:
            auto_x_mm, auto_y_mm = _auto_measurement_area_mm(
                workflow_config.measured_file_path
            )
            if area_x_mm is None:
                area_x_mm = auto_x_mm
            if area_y_mm is None:
                area_y_mm = auto_y_mm
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning(
                "Could not auto-derive measurement area from %s: %s",
                workflow_config.measured_file_path,
                exc,
            )
    measurement_area_x = f"{area_x_mm:g}" if area_x_mm is not None else "---"
    measurement_area_y = f"{area_y_mm:g}" if area_y_mm is not None else "---"
    pssar_measured = f"{workflow_result.measured_peak_wkg:.2f}"
    pssar_ref = f"{workflow_result.reference_pssar:.2f}"
    pssar_meas = f"{workflow_result.measured_pssar:.2f}"
    err_scale = f"{100.0 * workflow_result.scaling_error:.2f}"
    delta_dist = rf"{workflow_result.distance_to_agreement:g}~mm"
    delta_dose = rf"{workflow_result.dose_to_agreement:g}~\%"
    pass_rate = workflow_result.pass_rate_percent

    # Resolve pass/fail conditional
    if pass_rate < 100.0:
        fail_rate = f"{100.0 - pass_rate:.1f}"
        gamma_statement = (
            rf"The pattern validation fails because $\Gamma (x_e,y_e)~>~1.0$ "
            rf"at {fail_rate}\,\% of the locations of the measured sSAR "
            rf"distribution compared to the reference, "
        )
    else:
        gamma_statement = (
            r"The pattern validation passes because $\Gamma (x_e,y_e)~\leq~1.0$ "
            r"at all locations of the measured sSAR distribution, "
            r"compared to the reference, "
        )

    # Scaling error statement.
    # u_mr (antenna measurement uncertainty, in %) and pssar_criteria
    # (combined tolerance, in %) are computed here using the same formulas
    # as the Voila GUI (see voila.ipynb::_update_analytical_results).
    err_scale_abs = abs(100.0 * workflow_result.scaling_error)
    u_mr = 14.3 if antenna_type.upper() == "VPIFAS" else 9.4
    dose_da = float(workflow_result.dose_to_agreement)
    pssar_criteria = ((30.0 - dose_da) ** 2 + u_mr**2) ** 0.5
    pssar_criteria_str = f"{pssar_criteria:.1f}"
    u_mr_str = f"{u_mr:g}"
    err_scale_tolerance = "25.0"
    scale_statement_post = (
        rf"the tolerance of $\pm~\sqrt{{{err_scale_tolerance}~\%^2 + U_{{r,m}}^2}} "
        rf"= {pssar_criteria_str}~\%$, where $U_{{r,m}}^2$ = {u_mr_str}~\% "
        rf"for this antenna"
    )
    if err_scale_abs > pssar_criteria:
        scale_statement = (
            rf"The scaling error for the psSAR is outside {scale_statement_post}."
        )
    else:
        scale_statement = (
            rf"The scaling error for the psSAR is within {scale_statement_post}."
        )

    # Build the LaTeX snippet for this test case
    content = rf"""
\clearpage
\FloatBarrier
\begin{{center}}
    \section*{{SAR Pattern Assessment Report for IEC/IEEE PAS 62209-5}}
    \today
\end{{center}}

\subsection*{{Measured sSAR Parameters}}

File name: \texttt{{{measured_filename}}}\\
Measurement area: ($x$, $y$) = ({measurement_area_x}~mm, {measurement_area_y}~mm).

\begin{{table}}[htpb] \centering
\begin{{tabular}}{{ccccc||cccc}}

\textbf{{Source}} &\textbf{{Freq.}} & \textbf{{Dist.}} & \textbf{{Avg.}}& \textbf{{Noise}}& \textbf{{psSAR}} &\multicolumn{{2}}{{c}}{{\textbf{{psSAR at 30~dBm}}}}& \textbf{{Scaling}} \\
\textbf{{Type}} & & & \textbf{{Mass}} & \textbf{{Floor}}& \textbf{{Meas.}}& \textbf{{Meas.}} & \textbf{{Ref.}} & \textbf{{Error}} \\
& \textbf{{(MHz)}} & \textbf{{(mm)}} & \textbf{{(g)}} & \textbf{{(W/kg)}} &\textbf{{(W/kg)}} & \textbf{{(W/kg)}} & \textbf{{(W/kg)}} & \textbf{{(\%)}} \\\hline
{antenna_type} & {frequency_mhz} & {distance_mm} & {mass_g} & {noise_level} & {pssar_measured} & {pssar_meas} & {pssar_ref} & {err_scale} \\
\end{{tabular}}
\end{{table}}

\FloatBarrier
\subsection*{{Results}}

{gamma_statement} ~according to the Gamma criterion described in IEC/IEEE PAS 62209-5
with $\Delta D~=~${delta_dose}, $\Delta d$~=~{delta_dist}. {scale_statement}

\begin{{figure}}[h!]
  \centering
  \begin{{tabular}}{{c}}
    \includegraphics[width=.52\linewidth]{{{figures_relpath}/gamma_failures.png}}
  \end{{tabular}}%
  \begin{{tabular}}{{c}}
    \includegraphics[width=.26\linewidth]{{{figures_relpath}/gamma_index_with_colorbar.png}} \\
    \includegraphics[width=.26\linewidth]{{{figures_relpath}/registration_nocolorbar.png}} \\
  \end{{tabular}}
  \begin{{tabular}}{{c}}
    \includegraphics[width=.38\linewidth]{{{figures_relpath}/measured_with_colorbar.png}}
  \end{{tabular}}%
  \begin{{tabular}}{{c}}
    \includegraphics[width=.38\linewidth]{{{figures_relpath}/reference_with_colorbar.png}}
  \end{{tabular}}
\end{{figure}}
"""
    return content


def generate_report(
    *,
    workflow_result: WorkflowResult,
    workflow_config: WorkflowConfig,
    output_dir: str | Path,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    antenna_type: str = "dipole",
    frequency_mhz: int = 0,
    distance_mm: int = 0,
    mass_g: int = 0,
    compile_pdf: bool = True,
) -> Path:
    """
    Render or append a tested-case page to the SAR Pattern Validation report.

    On the first call (no existing main.tex in *output_dir*), the
    tested_case_report_page_new template is copied into *output_dir* and the
    first test case is inserted before \\end{document}.

    On subsequent calls (main.tex already exists), the new test case is
    appended before \\end{document}.

    When ``compile_pdf=True`` (default) and ``pdflatex`` is on PATH, compiles
    the .tex to PDF and returns the PDF path.  If pdflatex is absent the .tex
    path is returned instead (graceful degradation).

    Returns the path to the compiled PDF, or to ``main.tex`` when PDF
    compilation is unavailable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template_dir = Path(template_dir)

    out_path = output_dir / "main.tex"

    # --- Initialize from tested_case_report_page_new if this is the first run ---
    if not out_path.is_file():
        _initialize_report(output_dir, template_dir)

    # --- Determine case number and create figures directory ---
    case_num = _next_case_number(output_dir)
    case_figures_dir = output_dir / "figures" / f"case_{case_num:03d}"
    case_figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy workflow result figures into case-specific directory ---
    for target_name, attr_name in TEMPLATE_FIGURE_MAPPING.items():
        source_path = getattr(workflow_result, attr_name, None)
        if source_path is None or not Path(source_path).is_file():
            continue
        shutil.copy2(source_path, case_figures_dir / target_name)

    # --- Render the test case content ---
    figures_relpath = f"figures/case_{case_num:03d}"
    test_case_content = _render_test_case_body(
        workflow_result=workflow_result,
        workflow_config=workflow_config,
        antenna_type=antenna_type,
        frequency_mhz=frequency_mhz,
        distance_mm=distance_mm,
        mass_g=mass_g,
        figures_relpath=figures_relpath,
    )

    # --- Insert the test case before \end{document} in main.tex ---
    text = out_path.read_text(encoding="utf-8")
    if _DOCUMENT_END_MARKER not in text:
        raise ValueError(
            f"Cannot find '{_DOCUMENT_END_MARKER}' in {out_path}. "
            "The report template may be malformed."
        )
    text = text.replace(
        _DOCUMENT_END_MARKER,
        test_case_content + "\n" + _DOCUMENT_END_MARKER,
    )
    out_path.write_text(text, encoding="utf-8")

    # --- Compile PDF ---
    if compile_pdf:
        pdf_path = compile_report(out_path)
        if pdf_path is not None:
            return pdf_path

    return out_path
