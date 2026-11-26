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


def _label(name: str) -> str:
    """Sentence-case an output key without lowercasing units such as 50C."""
    text = _text(name.replace("_", " "))
    return text[:1].upper() + text[1:]


def _with_unit(value: object, unit: object) -> str:
    if value is None:
        return "—"
    return (
        f"{_text(value)} °{_text(unit)}"
        if unit == "C"
        else f"{_text(value)} {_text(unit)}"
    )


def _mean_sd(point: Mapping[str, Any]) -> str:
    mean, sd = point.get("measured_mean"), point.get("measured_std")
    if mean is None:
        return "—"
    unit = "°C" if point.get("unit") == "C" else _text(point.get("unit"))
    return f"{mean:.4g} ± {sd:.2g} {unit}"


def _span(point: Mapping[str, Any]) -> str:
    low, high = point.get("measured_min"), point.get("measured_max")
    if low is None or high is None:
        return "—"
    unit = "°C" if point.get("unit") == "C" else _text(point.get("unit"))
    return f"{low:.4g} – {high:.4g} {unit}"


def _minus_one(value: object) -> object:
    return value - 1 if isinstance(value, int) else value


def _total(values: object) -> object:
    if not isinstance(values, (list, tuple)) or not values:
        return None
    try:
        return float(sum(float(value) for value in values if value is not None))
    except (TypeError, ValueError):
        return None


def _range(values: object) -> str:
    if isinstance(values, (list, tuple)) and len(values) == 2:
        low, high = values
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            return f"{low:.4g} – {high:.4g}"
    return "—"


def _processing_row(point: Mapping[str, Any]) -> tuple[object, ...]:
    """One data point's integration, overlap-graph and validation summary."""
    integrity = point.get("integrity_checks", {})
    graph = point.get("position_scale_refinement", {})
    held_out = "—"
    folds = point.get("held_out_validation", [])
    if folds:
        metrics = {
            metric.get("label"): metric.get("absolute_overlap_percent")
            for metric in folds[0].get("metrics", [])
        }
        held_out = (
            f"{_text(metrics.get('physical_only'))} → "
            f"{_text(metrics.get('detector_only'))}"
        )
    rms = "—"
    if graph:
        rms = (
            f"{_text(graph.get('weighted_edge_rms_before'))} → "
            f"{_text(graph.get('weighted_edge_rms_after'))}"
        )
    return (
        point.get("label"),
        _range(point.get("theta_range_deg")),
        _range(point.get("output_theta_range_deg")),
        f"{_text(integrity.get('supported_bins'))} / "
        f"{_text(integrity.get('unsupported_bins'))}",
        f"{_text(graph.get('n_positions'))} / {_text(graph.get('n_edges'))} / "
        f"{_text(graph.get('n_components'))}",
        rms,
        held_out,
        integrity.get("status"),
        point.get("raw_audit", {}).get("status"),
        len(point.get("warnings", [])),
    )


def _point_summary(
    points: Sequence[Mapping[str, Any]],
    section: Any,
    key: str = "status",
) -> str:
    """Collapse a per-point check to one value, or list differing points."""
    values = [_text(section(point).get(key, "not yet recorded")) for point in points]
    if len(set(values)) == 1:
        return f"{values[0]} (all {len(points)} points)"
    return "; ".join(
        f"{_text(point.get('label'))}: {value}"
        for point, value in zip(points, values, strict=True)
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
    points = record.get("data_points") or []
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
    run_rows: list[tuple[str, object]] = [
        ("Source scan", inputs.get("scan", args.get("scan"))),
        ("Geometry source", inputs.get("geometry_source")),
        ("Geometry calibration", inputs.get("geometry", args.get("geometry"))),
        ("Started (UTC)", record.get("started_utc")),
        ("Finished (UTC)", record.get("finished_utc")),
        ("Elapsed (seconds)", record.get("elapsed_seconds")),
        ("Latest stage", record.get("current_stage")),
    ]
    if points:
        run_rows += [
            ("Data-point axis", inputs.get("data_point_axis")),
            ("Data points", len(points)),
            ("Frame partition", inputs.get("data_point_partition")),
            ("Raw frames", inputs.get("raw_frames")),
        ]
    else:
        run_rows += [
            ("Raw frames → analysis positions", frames),
            ("Processing 2theta range (degrees)", record.get("theta_range_deg")),
            ("Exported 2theta range (degrees)", record.get("output_theta_range_deg")),
        ]
    run_rows += [
        ("Output Q limit (Å⁻¹)", record.get("output_q_max_angstrom_inverse")),
        ("Output Q limit source", record.get("output_q_max_source")),
    ]
    if not points:
        run_rows += [
            ("Supported / unsupported bins", bins),
            ("Overlap graph: positions / edges / components", graph_size),
        ]
    table(["Run", "Value"], run_rows)

    if points:
        section("Data points")
        paragraph(
            "Each data point is a complete 2theta sweep processed independently "
            "(own detector fit, overlap scales and validation) into its own "
            "diffraction pattern. Temperatures are the recorded "
            f"{_text(inputs.get('data_point_axis'))} readings for that point's frames."
        )
        table(
            [
                "Point",
                "Setpoint",
                "Measured mean ± SD",
                "Measured min – max",
                "Raw frames (indices)",
                "Positions",
                "Summed I0",
                "Status",
            ],
            [
                (
                    point.get("label"),
                    _with_unit(point.get("setpoint"), point.get("unit")),
                    _mean_sd(point),
                    _span(point),
                    f"{_text(point.get('raw_frames'))} "
                    f"({_text(point.get('start'))}–"
                    f"{_text(_minus_one(point.get('stop')))})",
                    point.get("aggregation", {}).get("positions"),
                    _total(point.get("aggregation", {}).get("summed_i0")),
                    point.get("status"),
                )
                for point in points
            ],
        )
        table(
            [
                "Point",
                "Processing 2theta (°)",
                "Exported 2theta (°)",
                "Supported / unsupported bins",
                "Graph positions / edges / components",
                "Edge log-ratio RMS before → after",
                "Held-out absolute overlap (%) physical → detector",
                "Integrity",
                "Raw audit",
                "Warnings",
            ],
            [_processing_row(point) for point in points],
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
    integrity_status = (
        _point_summary(points, lambda point: point.get("integrity_checks", {}))
        if points
        else integrity.get("status", "not yet recorded")
    )
    constant_status = (
        _point_summary(
            points, lambda point: point.get("geometry_checks", {}), "constant_pattern"
        )
        if points
        else record.get("geometry_checks", {}).get(
            "constant_pattern", "not yet recorded"
        )
    )
    audit_status = (
        _point_summary(points, lambda point: point.get("raw_audit", {}))
        if points
        else record.get("raw_audit", {}).get("status", "not yet recorded")
    )
    table(
        ["Check", "Result"],
        [
            ("Validation mode", args.get("validation")),
            ("Integrity checks", integrity_status),
            ("Constant-pattern normalization", constant_status),
            ("Independent raw-group audit", audit_status),
        ],
    )
    labelled = [
        (point.get("label"), fold)
        for point in points
        for fold in point.get("held_out_validation", [])
    ] or [(None, fold) for fold in record.get("held_out_validation", [])]
    point_column = ["Point"] if points else []
    if labelled:
        paragraph(
            "Held-out detector results (percent; smaller discrepancies indicate "
            "closer agreement, not certified accuracy):"
        )
        table(
            [
                *point_column,
                "Fold",
                "Correction",
                "Centered overlap (%)",
                "Absolute overlap (%)",
                "Higher-Q RMS (%)",
            ],
            [
                (
                    *([label] if points else []),
                    fold.get("fold"),
                    metric.get("label"),
                    metric.get("centered_overlap_percent"),
                    metric.get("absolute_overlap_percent"),
                    metric.get("higher_q_rms_percent"),
                )
                for label, fold in labelled
                for metric in fold.get("metrics", [])
            ],
        )
        table(
            [
                *point_column,
                "Fold",
                "Training positions",
                "Held-out positions",
                "Infer test gains from lower Q",
            ],
            [
                (
                    *([label] if points else []),
                    fold.get("fold"),
                    len(fold.get("train_positions", [])),
                    len(fold.get("test_positions", [])),
                    fold.get("validate_frame_gains"),
                )
                for label, fold in labelled
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
        label = _label(name)
        target = quote(str(filename), safe="/")
        lines.append(f"- {label}: [{_text(filename)}]({target})")
    if not outputs:
        lines.append("No completed exports recorded yet.")
    lines.append("")
    for point in points:
        point_outputs = point.get("outputs", {})
        if not point_outputs:
            continue
        lines.append(f"**{_text(point.get('label'))}**")
        lines.append("")
        for name, filename in point_outputs.items():
            label = _label(name)
            # The combined pattern is addressed as file::/entry; link the file.
            target = quote(str(filename).split("::")[0], safe="/")
            lines.append(f"- {label}: [{_text(filename)}]({target})")
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

    def figure_gallery(gallery: Mapping[str, Any], heading: str) -> None:
        for filename in gallery.get("figures", []):
            stem = Path(filename).stem
            title = titles.get(stem, stem.replace("_", " "))
            target = quote(str(filename), safe="/")
            paragraph(
                f"{heading} {_text(title)}\n\n"
                f"[{_text(filename)}]({target})\n\n"
                f"![{title}]({target})"
            )
            if stem == "overlap_final_abcd":
                paragraph(
                    "**Low-support spikes:** individual-position curves can spike "
                    "where very few pixels contribute, especially near coverage "
                    "edges. Panel A includes all supported bins; panel B applies "
                    "the diagnostic pixel-weight threshold to both the position "
                    "and its reference. Pixel weights describe geometric support, "
                    "not photon counts. Such spikes alone do not establish a "
                    "correction defect. Both panels use the final applied gains."
                )

    galleries = [
        (point.get("label"), point.get("diagnostics", {})) for point in points
    ] or [(None, diagnostic)]
    for point_label, gallery in galleries:
        if point_label is not None:
            paragraph(f"### {_text(point_label)}")
        figure_gallery(gallery, "####" if point_label is not None else "###")
    if not any(gallery.get("figures") for _, gallery in galleries):
        paragraph("No completed diagnostic gallery recorded yet.")
    skips = [
        (point_label, item)
        for point_label, gallery in galleries
        for item in gallery.get("skips", [])
    ]
    if skips:
        table(
            [*point_column, "Skipped diagnostic", "Reason"],
            [
                (*([label] if points else []), item.get("plot"), item.get("reason"))
                for label, item in skips
            ],
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
