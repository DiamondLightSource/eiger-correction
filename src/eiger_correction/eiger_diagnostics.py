"""Headless, read-only diagnostics for the EIGER extraction CLI.

No detector files are opened and no fitting or integration is performed here.
Sampling must be supplied by the caller's integration pass: integer counts of
count-valid pixel centres, NOT pyFAI fractional pixel-splitting count weights.
Overlap statistics are descriptive, in-sample quantities, not uncertainties.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eiger_workflow as workflow

__all__ = ["save_diagnostics", "position_overlap_diagnostics"]


def position_overlap_diagnostics(group, min_pixel_weight=100):
    """Return notebook-equivalent leave-one-position-out B/E diagnostics.

    ``per_frame`` has shape (positions, 4, bins), ordered as signal,
    normalization, variance, pixel weight. ``accumulators`` contains the totals.
    B is 100 * (position intensity / all-other-position intensity - 1).
    E (influence) is B times the position's actual normalization fraction.
    Unsupported residuals/influences are NaN; unsupported fractions are zero.
    No expensive accumulator or influence identity assertions are repeated.
    """
    if not np.isfinite(min_pixel_weight) or min_pixel_weight <= 0:
        raise ValueError("min_pixel_weight must be finite and positive")
    frames = np.asarray(group["per_frame"], dtype=float)
    totals = np.asarray(group["accumulators"], dtype=float)
    if (frames.ndim != 3 or frames.shape[1] != 4
            or totals.shape != frames.shape[1:]):
        raise ValueError("Expected per_frame (positions, 4, bins) and accumulators (4, bins)")
    signal, norm, pixels = (frames[:, k, :] for k in (0, 1, 3))
    curves = np.divide(signal, norm, out=np.full_like(signal, np.nan), where=norm > 0)
    fraction = np.divide(norm, totals[1], out=np.zeros_like(norm), where=totals[1] > 0)
    other_norm = totals[1][None, :] - norm
    other_intensity = np.divide(
        totals[0][None, :] - signal, other_norm,
        out=np.full_like(signal, np.nan), where=other_norm > 0,
    )
    valid = ((norm > 0) & (other_norm > 0) & (pixels >= min_pixel_weight)
             & (totals[3][None, :] - pixels >= min_pixel_weight)
             & np.isfinite(curves) & np.isfinite(other_intensity) & (other_intensity > 0))
    residual = np.divide(
        100 * (curves - other_intensity), other_intensity,
        out=np.full_like(signal, np.nan), where=valid,
    )
    return {"curves": curves, "fraction": fraction, "residual": residual,
            "influence": fraction * residual, "valid": valid}


@contextmanager
def _figure(directory, summary, name, dpi, **kwargs):
    """Close even when drawing or JPEG encoding fails; register only saved files."""
    fig = plt.figure(constrained_layout=True, **kwargs)
    try:
        yield fig
        filename = name + ".jpg"
        fig.savefig(directory / filename, format="jpg", dpi=dpi)
        summary["figures"].append(filename)
    finally:
        plt.close(fig)


def _skip(summary, plot, reason):
    summary["skips"].append({"plot": plot, "reason": reason})
    summary["warnings"].append(f"{plot}: {reason}")


def _limits(value, name):
    limits = np.asarray(value, dtype=float)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
        raise ValueError(f"{name} must contain two increasing finite bounds")
    return tuple(map(float, limits))


def _grouping_plot(directory, summary, raw_scan, scan, groups, dpi):
    raw_angles = np.asarray(raw_scan.angles, dtype=float)
    exact, counts = np.unique(raw_angles, return_counts=True)
    minima = np.array([raw_angles[g].min() for g in groups])
    means = np.array([raw_angles[g].mean() for g in groups])
    spreads = np.array([np.ptp(raw_angles[g]) for g in groups])
    sizes = np.array([len(g) for g in groups])
    numbers = np.arange(1, len(groups) + 1)
    summary["grouping"] = {
        "raw_frames": int(raw_angles.size), "exact_angles": int(exact.size),
        "analysis_positions": len(groups), "group_sizes": sizes.tolist(),
        "maximum_span_deg": float(spreads.max()),
    }
    with _figure(directory, summary, "angle_grouping", dpi, figsize=(15, 9)) as fig:
        grid = fig.add_gridspec(2, 2)
        top = fig.add_subplot(grid[0, :])
        offsets = fig.add_subplot(grid[1, 0])
        zoom = fig.add_subplot(grid[1, 1])
        top.vlines(exact, 0, counts, color="0.7")
        top.scatter(exact, counts, s=15, color="0.4", label="Frames at exact angle")
        top.plot(scan.angles, sizes, "o", label="Actual analysis groups")
        top.set(xlabel="Motor 2θ (degrees)", ylabel="Raw frames", title="A · Actual frame grouping")
        top.legend()
        # Membership comes solely from the supplied aggregation, never reclustered.
        for i, members in enumerate(groups):
            values, repeats = np.unique(raw_angles[members], return_counts=True)
            offsets.scatter(np.full(values.size, i + 1), 1000 * (values - minima[i]),
                            s=15 + 30 * repeats / counts.max(), color="0.35", alpha=0.7)
        offsets.vlines(numbers, 0, 1000 * spreads, color="0.6")
        offsets.scatter(numbers, 1000 * (means - minima), marker="D", s=18, color="tab:orange")
        offsets.set(title="B · Within-group motor spread", xlabel="Actual group (1-based)",
                    ylabel="Motor angle − group minimum (mdeg)")
        widest = int(np.argmax(spreads))
        values, repeats = np.unique(raw_angles[groups[widest]], return_counts=True)
        zoom.vlines(1000 * (values - minima[widest]), 0, repeats, color="0.4")
        zoom.scatter(1000 * (values - minima[widest]), repeats)
        zoom.axvline(1000 * (means[widest] - minima[widest]), color="tab:orange", linestyle=":")
        zoom.set(title=f"C · Widest group {widest + 1}: {sizes[widest]} frames",
                 xlabel="Motor angle − group minimum (mdeg)", ylabel="Exact-angle repeats")
        for ax in (top, offsets, zoom):
            ax.grid(alpha=0.2)
        fig.suptitle("Recorded motor readings and actual analysis-group membership")


def _azimuth_plot(directory, summary, scan, model, base_mask, dpi):
    mask = np.asarray(base_mask, dtype=bool)
    denominator = np.count_nonzero(~mask)
    selected = (int(np.argmin(scan.angles)), int(np.argmax(scan.angles)))
    previews = {}
    retained = []
    for i, angle in enumerate(scan.angles):
        kept = ~(mask | workflow.azimuth_mask(model, float(angle)))
        retained.append(int(np.count_nonzero(kept)))
        if i in selected:
            previews[i] = kept
    summary["azimuth"] = {"retained_pixels_by_position": retained,
                           "fixed_unmasked_pixels": int(denominator)}
    with _figure(directory, summary, "azimuth_geometry", dpi, figsize=(15, 4)) as fig:
        axes = fig.subplots(1, 3)
        for ax, i in zip(axes[:2], selected):
            ax.imshow(previews[i], origin="lower", aspect="auto", cmap="gray", vmin=0, vmax=1)
            ax.set(title=f"Motor {scan.angles[i]:.3f}° · white = retained",
                   xlabel="Detector column", ylabel="Detector row")
        if denominator:
            axes[2].plot(scan.angles, 100 * np.asarray(retained) / denominator, ".-")
        else:
            axes[2].text(0.5, 0.5, "All pixels fixed-masked", ha="center", transform=axes[2].transAxes)
        axes[2].set(xlabel="Motor position (degrees)", ylabel="Fixed-unmasked pixels retained (%)",
                    ylim=(0, 105), title=f"Chi wedge: {getattr(model, 'azimuth_range', None)}°")
        fig.suptitle("Per-position azimuth mask · geometry/pixel centres only, before count rejection")


def _ring_current(scan, raw_scan, groups):
    """Best-effort optional metadata; malformed/missing monitors are harmless."""
    try:
        ring = getattr(scan, "ring_current", None)
        if ring is None:
            raw_ring = getattr(raw_scan, "ring_current", None)
            if raw_ring is None:
                return None
            raw_ring = np.asarray(raw_ring, dtype=float)
            if raw_ring.shape != np.asarray(raw_scan.angles).shape:
                return None
            ring = np.array([raw_ring[g].mean() for g in groups])
        ring = np.asarray(ring, dtype=float)
        if ring.shape == np.asarray(scan.angles).shape and np.all(np.isfinite(ring) & (ring > 0)):
            return ring
    except (TypeError, ValueError, IndexError, OverflowError):
        pass
    return None


def _fit_plot(directory, summary, raw_scan, scan, groups, fit, dpi):
    indices = np.asarray(fit.get("fit_indices", np.arange(len(scan.angles))), dtype=int)
    with _figure(directory, summary, "detector_response", dpi, figsize=(15, 10)) as fig:
        axes = fig.subplots(2, 2)
        for ax, field, title in ((axes[0, 0], "pixel_divisor", "Pixel-response divisor"),
                                 (axes[0, 1], "combined_divisor", "Pixel × sensor divisor")):
            image = ax.imshow(fit[field], origin="lower", aspect="auto", cmap="coolwarm", vmin=.95, vmax=1.05)
            ax.set(title=title, xlabel="Detector column", ylabel="Detector row")
            fig.colorbar(image, ax=ax)
        sensors = np.asarray(fit["sensor_gains"])
        axes[1, 0].plot(np.arange(1, sensors.size + 1), sensors, "o-")
        axes[1, 0].set(title="Sensor gains", xlabel="Sensor region", ylabel="Divisor gain")
        ax = axes[1, 1]
        ax.plot(np.asarray(scan.angles)[indices], np.asarray(fit["frame_gains"])[indices],
                ".-", label="Stage-1 position gain")
        i0 = np.asarray(scan.i0, dtype=float)[indices]
        if i0.size and np.all(np.isfinite(i0)) and i0.mean() > 0:
            ax.plot(np.asarray(scan.angles)[indices], i0 / i0.mean(), ".-", label="Summed group I0 / mean")
        # Optional already-loaded metadata only; never reopen a NeXus/detector file.
        ring = _ring_current(scan, raw_scan, groups)
        if ring is not None:
            values = ring[indices]
            if values.size:
                ax.plot(np.asarray(scan.angles)[indices], values / values.mean(), ".--",
                        label="Mean group ring current / mean")
        else:
            _skip(summary, "ring_current_overlay", "No valid preloaded ring_current metadata (optional)")
        ax.axhline(1, color="black", linewidth=.7)
        ax.set(title="Stage-1 gains and measured monitor diagnostics", xlabel="Motor position (degrees)",
               ylabel="Gain / normalized monitor")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.legend(fontsize=8)


def _sampling_plot(directory, summary, sampling, model, n_positions, dpi):
    if sampling is None:
        _skip(summary, "sampling_q", "No pixel-centre sampling supplied; detector data are not reread")
        return
    observations = np.asarray(sampling["observations_by_position"])
    sizes = np.asarray(sampling["group_sizes"])
    edges = np.asarray(sampling["theta_edges"], dtype=float)
    if (observations.ndim != 2 or observations.shape[0] != n_positions
            or observations.shape[1] == 0 or observations.dtype.kind not in "iu"
            or np.any(observations < 0) or sizes.shape != (n_positions,)
            or not np.all(np.isfinite(sizes) & (sizes >= 1) & (sizes == np.floor(sizes)))
            or edges.shape != (observations.shape[1] + 1,)
            or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0)
            or edges[0] < 0 or edges[-1] > 180):
        _skip(summary, "sampling_q", "Invalid integer pixel-centre sampling payload")
        return
    wavelength = float(model.geometry.wavelength) * 1e10
    if not np.isfinite(wavelength) or wavelength <= 0:
        _skip(summary, "sampling_q", "Wavelength must be finite and positive")
        return

    def to_q(theta):
        return 4 * np.pi * np.sin(np.deg2rad(theta) / 2) / wavelength

    def to_theta(q):
        return 2 * np.rad2deg(np.arcsin(np.clip(q * wavelength / (4 * np.pi), -1, 1)))

    total = observations.sum(axis=0)
    equivalents = sizes.astype(np.int64) @ observations.astype(np.int64, copy=False)
    coverage = np.count_nonzero(observations, axis=0)
    summary["sampling"] = {"observations": int(total.sum()),
                           "raw_frame_equivalents": int(equivalents.sum()),
                           "empty_bins": int(np.count_nonzero(total == 0)),
                           "bins": int(total.size), "theta_range_deg": edges[[0, -1]].tolist(),
                           "source": "count-valid pixel-centre histogram; not pyFAI count weights"}
    with _figure(directory, summary, "sampling_q", dpi, figsize=(15, 5)) as fig:
        axes = fig.subplots(1, 2)
        q_edges = to_q(edges)
        axes[0].stairs(total, q_edges, label="Retained pixel × analysis-position observations")
        if np.any(sizes > 1):
            axes[0].stairs(equivalents, q_edges, label="Raw-frame equivalents (approx. group geometry)")
            axes[0].set_yscale("symlog", linthresh=1)
        axes[0].set(title="A · Count-valid observations · final θ bins", ylabel="Pixel observations")
        axes[0].legend(fontsize=8)
        axes[1].stairs(coverage, q_edges, fill=True, alpha=.6, color="tab:green")
        axes[1].axhline(n_positions, color="0.4", linestyle="--")
        axes[1].set(title="B · Analysis-position coverage", ylabel="Positions contributing ≥1 pixel",
                    ylim=(0, 1.1 * max(1, n_positions)))
        for ax in axes:
            ax.set(xlabel=r"Q ($\mathrm{\AA}^{-1}$)", xlim=(q_edges[0], q_edges[-1]))
            ax.secondary_xaxis("top", functions=(to_theta, to_q)).set_xlabel("2θ (degrees)")
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Full-range sampling after final masks and count rejection · nonlinear Q edges")


def _residual_summary(diagnostics, window, angles):
    rows = []
    for i, angle in enumerate(angles):
        values = diagnostics["residual"][i, window & diagnostics["valid"][i]]
        rows.append({"position": i, "motor_deg": float(angle), "overlap_bins": int(values.size),
                     "mean_percent": float(values.mean()) if values.size else None,
                     "rms_percent": float(np.sqrt(np.mean(values ** 2))) if values.size else None})
    return rows


def _decorate(axes, diagnostic_range, boundaries):
    for ax in axes:
        ax.set_xlim(*diagnostic_range)
        ax.grid(alpha=.2)
        for boundary in boundaries:
            if diagnostic_range[0] <= boundary <= diagnostic_range[1]:
                ax.axvline(boundary, color="0.4", linestyle=":", linewidth=1)


def _overlap_plots(directory, summary, group, final, angles, window,
                   diagnostic_range, colors, boundaries, dpi):
    x = np.asarray(group["axis"])
    intensity = np.asarray(group["intensity"])
    # Keep the existing filename so links in previously generated reports work.
    with _figure(directory, summary, "overlap_final_abcd", dpi, figsize=(15, 8)) as fig:
        axes = fig.subplots(2, 1, sharex=True)
        for i in range(len(angles)):
            for ax, field in zip(axes[:2], ("curves", "residual")):
                ax.plot(x[window], final[field][i, window], color=colors[i], alpha=.7, linewidth=.7)
        axes[0].plot(x[window], intensity[window], color="black", linewidth=1.5, label="Final merge")
        axes[0].set(title="A · Individual-position curves (including low pixel support) and final merge",
                ylabel="Intensity (a.u.)")
        axes[0].legend()
        axes[1].axhline(0, color="black", linewidth=.8)
        axes[1].set(title="B · Position minus all OTHER positions · pixel-support filtered",
                    xlabel="2θ (degrees)", ylabel="Overlap residual (%)")
        _decorate(axes, diagnostic_range, boundaries)
        fig.suptitle("After final applied gains · measured I0 retained; dotted lines: detector-fit limits")
        fig.supxlabel(
            "Individual-position curves can spike where very few pixels contribute, especially near coverage edges.\n"
            "A shows all supported bins; B applies the diagnostic pixel-weight threshold to both position and reference.\n"
            "Pixel-splitting weights are geometric support, not photon counts. Spikes are not by themselves evidence of a correction defect.\n"
            "Residuals are in-sample diagnostics, not independent validation or uncertainty estimates.",
            fontsize=9,
        )


def _gains_plot(directory, summary, scan, fit, processed_final, dpi):
    angles = np.asarray(scan.angles)
    stage1 = np.asarray(fit["frame_gains"])
    final = np.asarray(processed_final["frame_gains"])
    with _figure(directory, summary, "position_gains", dpi, figsize=(13, 4)) as fig:
        ax = fig.subplots()
        ax.plot(angles, stage1, ".-", label="Stage 1 nuisance gain")
        ax.plot(angles, final, ".-", label="Final applied divisor")
        if "fit_indices" in fit:
            unfitted = ~np.isin(np.arange(len(angles)), fit["fit_indices"])
            ax.scatter(angles[unfitted], stage1[unfitted], marker="x", color="black",
                       label="No stage-1 fitting observations")
        ax.axhline(1, color="0.4", linewidth=.8)
        ax.set(xlabel="Motor position (degrees)", ylabel="Intensity divisor",
               title="Final scales replace, rather than multiply, stage-1 scales")
        ax.grid(alpha=.2)
        ax.legend()


def _xy_plot(directory, summary, group, theta_range, diagnostic_range, dpi):
    x = np.asarray(group["axis"])
    intensity = np.asarray(group["intensity"])
    supported = ((np.asarray(group["accumulators"])[1] > 0)
                 & np.isfinite(x) & np.isfinite(intensity))
    zoom = supported & (x >= diagnostic_range[0]) & (x <= diagnostic_range[1])
    with _figure(directory, summary, "final_xy", dpi, figsize=(13, 8)) as fig:
        axes = fig.subplots(2, 1)
        # Keep unsupported bins as gaps rather than interpolate across them.
        axes[0].plot(x, np.where(supported, intensity, np.nan), linewidth=1)
        axes[0].set(title="A · Full final integrated angular range", xlim=theta_range)
        axes[0].axvspan(*diagnostic_range, color="tab:orange", alpha=.12, label="Zoom interval")
        axes[0].legend()
        if zoom.any():
            axes[1].plot(x, np.where(zoom, intensity, np.nan), linewidth=1)
        else:
            axes[1].text(.5, .5, "No supported intensity in zoom interval", ha="center", transform=axes[1].transAxes)
            _skip(summary, "final_xy_zoom", "No supported intensity in diagnostic range; full-range plot saved")
        axes[1].set(title="B · High-angle zoom", xlim=diagnostic_range)
        for ax in axes:
            ax.set(xlabel="2θ (degrees)", ylabel="Normalized intensity (a.u.)")
            ax.grid(alpha=.25)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
        fig.suptitle("Final integrated intensities · measured I0 retained")
        fig.supxlabel("One scalar per position; no smoothing, tapering, background subtraction or Q-dependent scale.")


def save_diagnostics(output_directory: Path, raw_scan, scan, frame_groups, model,
                     base_mask, fit, processed_unity, processed_final, scale_fit, *,
                     theta_range, diagnostic_range=(30., 80.), min_pixel_weight=100,
                     dpi=140) -> dict:
    """Save JPEG diagnostics and return a strict-JSON-compatible summary.

    ``frame_groups`` lists raw-frame indices for each analysis position, in
    ``scan.angles`` order, including singleton groups when not aggregating.
    ``processed_unity`` supplies sampling metadata only. Curves and residuals
    use ``processed_final['groups']['merged_2theta']``, after the final gains.
    No scales are applied here. ``scale_fit`` may omit graph/RMS metadata.

    Optional ``processed_unity['sampling']`` contains observations_by_position
    (integer positions × bins), group_sizes, and theta_edges (degrees). It must
    be computed during the CLI's single integration pass after final masks and
    count validity checks, not reconstructed from pyFAI count accumulators.
    Optional preloaded ``ring_current`` on scan (group means) or raw_scan (raw
    readings) adds a monitor overlay; no file is opened to obtain it.

    Empty diagnostic windows/overlaps are reported in skips and warnings while
    independent plots are still saved. Invalid required inputs and filesystem
    errors propagate; figures close even on failure. Output names are stable
    and existing JPEGs in the chosen directory are replaced.
    """
    theta_range = _limits(theta_range, "theta_range")
    diagnostic_range = _limits(diagnostic_range, "diagnostic_range")
    if not np.isfinite(min_pixel_weight) or min_pixel_weight <= 0:
        raise ValueError("min_pixel_weight must be finite and positive")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be finite and positive")
    groups = [np.asarray(g, dtype=int) for g in frame_groups]
    if len(groups) != len(scan.angles) or not groups or any(g.ndim != 1 or not g.size for g in groups):
        raise ValueError("frame_groups must contain one nonempty index array per analysis position")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"figures": [], "skips": [], "warnings": [], "residuals": {},
               "theta_range_deg": list(theta_range), "diagnostic_range_deg": list(diagnostic_range),
               "min_pixel_weight": float(min_pixel_weight),
               "interpretation": "In-sample descriptive residuals; shared scales/calibration induce correlations. "
                                 "Pixel support is not a photon-count threshold; no scale uncertainty is propagated."}
    _grouping_plot(directory, summary, raw_scan, scan, groups, dpi)
    _azimuth_plot(directory, summary, scan, model, base_mask, dpi)
    _fit_plot(directory, summary, raw_scan, scan, groups, fit, dpi)
    _sampling_plot(directory, summary, processed_unity.get("sampling"), model, len(groups), dpi)
    _gains_plot(directory, summary, scan, fit, processed_final, dpi)
    group = processed_final["groups"]["merged_2theta"]
    _xy_plot(directory, summary, group, theta_range, diagnostic_range, dpi)

    # Inspect the final exported integration without reapplying any gains.
    final = position_overlap_diagnostics(group, min_pixel_weight)
    x = np.asarray(group["axis"], dtype=float)
    angles = np.asarray(scan.angles, dtype=float)
    window = (x >= diagnostic_range[0]) & (x <= diagnostic_range[1])
    summary["residuals"] = {"final": _residual_summary(final, window, angles)}
    summary["scale_fit"] = {}
    for key in ("n_positions", "n_edges", "n_components", "n_pair_bins", "edge_rms_before",
                "edge_rms_after", "weighted_edge_rms_before", "weighted_edge_rms_after"):
        value = (scale_fit or {}).get(key)
        if value is not None:
            summary["scale_fit"][key] = (int(value) if key.startswith("n_") else
                                         float(value) if np.isfinite(value) else None)
    if not window.any():
        reason = "Diagnostic range does not overlap the integrated grid"
    elif not np.any(final["valid"][:, window]):
        reason = "No final leave-one-position-out overlap passes the pixel-weight threshold"
    else:
        reason = None
    if reason is not None:
        _skip(summary, "overlap_final_abcd", reason)
        return summary
    colors = plt.get_cmap("turbo")(np.linspace(.03, .97, len(angles)))
    # Existing integration metadata identifies the detector-fit interval without
    # requiring another fit-range argument, geometry pass or Q integration.
    within_fit = np.asarray(group.get("within_fit_range", np.zeros(x.size, dtype=bool)), dtype=bool)
    boundaries = x[within_fit][[0, -1]] if within_fit.any() else []
    _overlap_plots(directory, summary, group, final, angles, window,
                   diagnostic_range, colors, boundaries, dpi)
    return summary