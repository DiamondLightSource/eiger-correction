"""Human-readable companion to the machine-readable extraction record."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote


def _text(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(item) for item in value)
    return (
        str(value)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_markdown(record: Mapping[str, Any]) -> str:
    """Render a JSON-safe record, including partial/failed runs, without I/O.

    Large machine-facing arrays, hashes and graph edges stay in report.json.
    Image paths are relative to the report so the directory remains portable.
    """
    lines = ["# EIGER extraction report", ""]

    def paragraph(text: str) -> None:
        lines.extend([text, ""])

    def section(title: str) -> None:
        lines.extend([f"## {title}", ""])

    def table(headers: Sequence[str], rows: Iterable[Iterable[object]]) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        lines.extend(
            "| " + " | ".join(_text(value) for value in row) + " |" for row in rows
        )
        lines.append("")

    args = record.get("arguments", {})
    inputs = record.get("inputs", {})
    aggregation = record.get("aggregation", {})
    integrity = record.get("integrity_checks", {})
    graph = record.get("position_scale_refinement", {})
    diagnostic = record.get("diagnostics", {})
    status = record.get("status", "unknown")
    paragraph(f"**Status: {_text(status).replace('_', ' ').upper()}**")
    if status == "running":
        paragraph(
            "This is a checkpoint, not a completed result. "
            "If the process has stopped, this record may be incomplete."
        )
    elif status in ("failed", "interrupted"):
        paragraph(
            "**Do not accept partial outputs from this run.** "
            "See the failure details below."
        )
    paragraph("[Machine-readable record](report.json) · [Full process log](run.log)")
    frames = f"{inputs.get('raw_frames', '—')} → {aggregation.get('positions', '—')}"
    bins = (
        f"{integrity.get('supported_bins', '—')} / "
        f"{integrity.get('unsupported_bins', '—')}"
    )
    graph_size = (
        f"{graph.get('n_positions', '—')} / {graph.get('n_edges', '—')} / "
        f"{graph.get('n_components', '—')}"
    )
    table(
        ["Run", "Value"],
        [
            ("Source scan", inputs.get("scan", args.get("scan"))),
            ("Geometry source", inputs.get("geometry_source")),
            ("Geometry calibration", inputs.get("geometry", args.get("geometry"))),
            ("Started (UTC)", record.get("started_utc")),
            ("Finished (UTC)", record.get("finished_utc")),
            ("Elapsed (seconds)", record.get("elapsed_seconds")),
            ("Latest stage", record.get("current_stage")),
            ("Raw frames → analysis positions", frames),
            ("Processing 2theta range (degrees)", record.get("theta_range_deg")),
            ("Exported 2theta range (degrees)", record.get("output_theta_range_deg")),
            ("Output Q limit (Å⁻¹)", record.get("output_q_max_angstrom_inverse")),
            ("Output Q limit source", record.get("output_q_max_source")),
            ("Supported / unsupported bins", bins),
            ("Overlap graph: positions / edges / components", graph_size),
        ],
    )

    section("Warnings and interpretation")
    messages = record.get("warnings", [])
    if messages:
        for message in messages:
            lines.append(f"- {_text(message)}")
        lines.append("")
    else:
        paragraph(
            "No warnings recorded so far."
            if status == "running"
            else "No warnings recorded."
        )
    paragraph(
        _text(
            record.get(
                "validation_scope",
                "Stage-2 overlap residuals are in-sample, not independent validation.",
            )
        )
    )
    for limitation in record.get("limitations", []):
        lines.append(f"- {_text(limitation)}")
    lines.append("")

    error = record.get("error")
    if error:
        section("Failure details")
        paragraph(f"**{_text(error.get('type'))}:** {_text(error.get('message'))}")
        # A variable fence keeps arbitrary exception text from closing the block.
        trace = str(error.get("traceback", ""))
        fence = "```"
        while fence in trace:
            fence += "`"
        paragraph(f"{fence}text\n{trace.rstrip()}\n{fence}")

    section("Validation")
    table(
        ["Check", "Result"],
        [
            ("Validation mode", args.get("validation")),
            ("Integrity checks", integrity.get("status", "not yet recorded")),
            (
                "Constant-pattern normalization",
                record.get("geometry_checks", {}).get(
                    "constant_pattern", "not yet recorded"
                ),
            ),
            (
                "Independent raw-group audit",
                record.get("raw_audit", {}).get("status", "not yet recorded"),
            ),
        ],
    )
    folds = record.get("held_out_validation", [])
    if folds:
        paragraph(
            "Held-out detector results (percent; smaller discrepancies indicate "
            "closer agreement, not certified accuracy):"
        )
        table(
            [
                "Fold",
                "Correction",
                "Centered overlap (%)",
                "Absolute overlap (%)",
                "Higher-Q RMS (%)",
            ],
            [
                (
                    fold.get("fold"),
                    metric.get("label"),
                    metric.get("centered_overlap_percent"),
                    metric.get("absolute_overlap_percent"),
                    metric.get("higher_q_rms_percent"),
                )
                for fold in folds
                for metric in fold.get("metrics", [])
            ],
        )
        table(
            [
                "Fold",
                "Training positions",
                "Held-out positions",
                "Infer test gains from lower Q",
            ],
            [
                (
                    fold.get("fold"),
                    len(fold.get("train_positions", [])),
                    len(fold.get("test_positions", [])),
                    fold.get("validate_frame_gains"),
                )
                for fold in folds
            ],
        )
    else:
        paragraph(
            "No held-out results recorded. Basic validation does not include "
            "independent detector folds; running/failed records may be incomplete."
        )
    if graph:
        table(
            ["Stage-2 descriptive quantity", "Value"],
            [
                (
                    "Weighted edge log-ratio RMS before",
                    graph.get("weighted_edge_rms_before"),
                ),
                (
                    "Weighted edge log-ratio RMS after",
                    graph.get("weighted_edge_rms_after"),
                ),
                ("Gauge", graph.get("gauge")),
            ],
        )
        paragraph(
            "These graph RMS values are in natural-log units, are in-sample "
            "and are not calibrated uncertainties."
        )

    section("Stage timings")
    table(
        ["Stage", "Status", "Seconds"],
        [
            (name.replace("_", " "), stage.get("status"), stage.get("seconds"))
            for name, stage in record.get("stages", {}).items()
        ],
    )

    section("Outputs")
    outputs = record.get("outputs", {})
    for name, filename in outputs.items():
        label = _text(name.replace("_", " ").capitalize())
        target = quote(str(filename), safe="/")
        lines.append(f"- {label}: [{_text(filename)}]({target})")
    if not outputs:
        lines.append("No completed exports recorded yet.")
    lines.append("")

    section("Diagnostic images")
    paragraph(
        "Images are embedded below; select the filename link to open the "
        "full-size JPEG. Relative links remain valid when this run directory "
        "is moved as a unit."
    )
    titles = {
        "angle_grouping": "Angle grouping and repeat counts",
        "azimuth_geometry": "Per-position azimuth mask — geometry only",
        "detector_response": "Detector response, sensor gains and monitors",
        "sampling_q": "Count-valid sampling and position coverage",
        "position_gains": "Stage-1 nuisance gains versus final position divisors",
        "final_xy": "Final intensity pattern — full range and high-angle zoom",
        "overlap_final_abcd": (
            "Position curves and overlap residuals after final gains (in-sample)"
        ),
    }
    for filename in diagnostic.get("figures", []):
        title = titles.get(Path(filename).stem, Path(filename).stem.replace("_", " "))
        target = quote(str(filename), safe="/")
        paragraph(
            f"### {_text(title)}\n\n"
            f"[{_text(filename)}]({target})\n\n"
            f"![{title}]({target})"
        )
        if Path(filename).stem == "overlap_final_abcd":
            paragraph(
                "**Low-support spikes:** individual-position curves can spike "
                "where very few pixels contribute, especially near coverage edges. "
                "Panel A includes all supported bins; panel B applies the "
                "diagnostic pixel-weight threshold to both the position and its "
                "reference. Pixel weights describe geometric support, not photon "
                "counts. Such spikes alone do not establish a correction defect. "
                "Both panels use the final applied gains."
            )
    if not diagnostic.get("figures"):
        paragraph("No completed diagnostic gallery recorded yet.")
    if diagnostic.get("skips"):
        table(
            ["Skipped diagnostic", "Reason"],
            [(item.get("plot"), item.get("reason")) for item in diagnostic["skips"]],
        )

    section("Processing settings")
    table(
        ["Option", "Value"],
        [(key.replace("_", " "), value) for key, value in args.items()],
    )
    software = record.get("software", {})
    if software:
        section("Software and reproducibility")
        paragraph(f"**Python:** {_text(software.get('python'))}")
        paragraph(f"**Platform:** {_text(software.get('platform'))}")
        table(["Package", "Version"], software.get("packages", {}).items())
    paragraph(
        "The JSON companion retains full fit history, per-position values, "
        "graph edges, residual summaries and provenance hashes without "
        "crowding this overview."
    )
    return "\n".join(lines)


def write_markdown_report(path: str | Path, record: Mapping[str, Any]) -> None:
    """Atomically replace only the human-readable companion report."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(render_markdown(record), encoding="utf-8")
    temporary.replace(path)
