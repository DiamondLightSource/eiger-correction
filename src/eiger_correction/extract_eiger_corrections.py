"""Headless, two-stage EIGER scan extraction with durable diagnostics and reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from numpy.typing import NDArray

from eiger_correction.eiger_report import write_markdown_report

if TYPE_CHECKING:
    from eiger_correction.eiger_workflow import (
        BoolArray,
        Payload,
        PhysicalModel,
        Scan,
    )


def json_safe(value: Any) -> Any:
    """Strict JSON: nonfinite scientific metrics are null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


class RunRecord:
    """Checkpoint the report at stage boundaries, including exceptions."""

    def __init__(self, directory: Path, arguments: argparse.Namespace) -> None:
        self.path = directory / "report.json"
        self.started = time.perf_counter()
        self.data: dict[str, Any] = {
            "schema": "eiger_extraction_report_v1",
            "status": "running",
            "started_utc": datetime.now(UTC).isoformat(),
            "arguments": vars(arguments),
            "stages": {},
            "warnings": [],
            "validation_scope": (
                "Held-out checks test stage 1 only. "
                "Stage-2 overlap residuals are in-sample."
            ),
            "limitations": [
                "Detector response outside the fit Q range is extrapolated.",
                "Poisson sigma excludes scale, calibration, I0 and model uncertainty "
                "and bin covariance.",
                "Source-position gains are source-scan-only, "
                "not transferable corrections.",
                "Successful integrity checks do not certify scientific accuracy.",
            ],
        }
        self.save()

    def save(self) -> None:
        self.data["elapsed_seconds"] = time.perf_counter() - self.started
        snapshot = json_safe(self.data)
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(snapshot, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(self.path)
        write_markdown_report(self.path.with_suffix(".md"), snapshot)

    @contextmanager
    def stage(self, name: str) -> Generator[None]:
        entry: dict[str, Any] = {"status": "running"}
        self.data["stages"][name] = entry
        self.data["current_stage"] = name
        self.save()
        started = time.perf_counter()
        try:
            yield
        except BaseException:
            entry["status"] = "failed"
            raise
        else:
            entry["status"] = "passed"
        finally:
            entry["seconds"] = time.perf_counter() - started
            self.save()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("scan", type=Path, help="Source NeXus scan (read-only)")
    result.add_argument(
        "--geometry",
        type=Path,
        help=(
            "Override goniometer calibration JSON; default: "
            "/entry/plan_metadata/geometry_calibration in the scan"
        ),
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; existing directories are never overwritten",
    )
    for name in ("detector", "monitor", "metadata"):
        result.add_argument(
            f"--{name}", type=Path, help=f"Override companion {name} file"
        )
    result.add_argument(
        "--azimuth-range",
        type=float,
        nargs=2,
        default=(67.5, 112.5),
        metavar=("LOW", "HIGH"),
    )
    result.add_argument(
        "--all-azimuths",
        action="store_true",
        help="Disable the default single chi wedge",
    )
    result.add_argument(
        "--fit-q-range",
        type=float,
        nargs=2,
        default=(10.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Detector-fit Q bounds in reciprocal angstroms (A^-1)",
    )
    result.add_argument(
        "--q-step-size",
        type=float,
        default=0.025,
        help="Target Q bin width in A^-1 for fitting and final extraction",
    )
    result.add_argument("--iterations", type=int, default=10)
    result.add_argument(
        "--two-theta-step-size",
        type=float,
        default=0.025,
        help="Target 2theta bin width in degrees",
    )
    result.add_argument(
        "--q-max",
        type=float,
        default=None,
        help=(
            "Final-output maximum Q in A^-1; default: unmasked pyFAI maximum. "
            "Does not limit detector fitting or overlap refinement"
        ),
    )
    result.add_argument("--angle-tolerance", type=float, default=0.001)
    result.add_argument(
        "--saturation",
        type=float,
        default=1e7,
        help="Raw-frame rejection threshold (not applied to valid sums)",
    )
    result.add_argument("--polarization", type=float, default=0.95)
    result.add_argument(
        "--density", type=float, default=6.2, help="CdTe density in g/cm^3"
    )
    result.add_argument("--overlap-min-pixels", type=float, default=100.0)
    result.add_argument("--overlap-min-bins", type=int, default=20)
    result.add_argument("--overlap-floor", type=float, default=0.003)
    result.add_argument(
        "--validation",
        choices=("basic", "holdout", "full"),
        default="holdout",
        help=(
            "Always integrity checks; holdout adds one independent detector fold, "
            "full adds both folds and lower-Q gain checks"
        ),
    )
    result.add_argument(
        "--raw-audit",
        action="store_true",
        help="Reread and reintegrate the highest-position raw group independently",
    )
    result.add_argument(
        "--diagnostic-range",
        type=float,
        nargs=2,
        default=(30.0, 80.0),
        metavar=("LOW", "HIGH"),
    )
    result.add_argument("--jpeg-dpi", type=int, default=140)
    return result


def load_geometry(
    scan_path: str | Path, geometry_path: str | Path | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return the selected geometry dictionary and its source/checksum.

    An explicit file always wins: file errors never silently fall back to NeXus.
    Hash the selected JSON bytes, not a reserialization of the decoded object.
    The existing PhysicalModel handles goniometer loading and compatibility checks.
    """
    if geometry_path is not None:
        path = Path(geometry_path).resolve()
        payload = path.read_bytes()
        provenance = {"geometry": str(path), "geometry_source": "file"}
    else:
        path = Path(scan_path).resolve()
        dataset_path = "/entry/plan_metadata/geometry_calibration"
        message = (
            f"No usable embedded geometry JSON at {path}::{dataset_path}; "
            "supply --geometry JSON_FILE"
        )
        with h5py.File(path, "r") as source:
            node = source.get(dataset_path)
            if not isinstance(node, h5py.Dataset):
                raise ValueError(message)
            value = np.asarray(node[()])
            if value.size != 1:
                raise ValueError(message + " (expected one JSON blob)")
            blob = value.reshape(-1)[0]
            if isinstance(blob, str):
                payload = blob.encode("utf-8")
            elif isinstance(blob, bytes):
                payload = bytes(blob)
            else:
                raise ValueError(message + " (expected a string or byte blob)")
        provenance = {
            "geometry": f"{path}::{dataset_path}",
            "geometry_source": "nexus",
            "geometry_dataset": dataset_path,
        }
    try:
        config = json.loads(payload)
    except (ValueError, UnicodeError) as error:
        raise ValueError(
            f"Invalid geometry JSON from {provenance['geometry']}; "
            "supply a valid --geometry JSON_FILE"
        ) from error
    if not isinstance(config, dict):
        raise ValueError(
            f"Geometry JSON from {provenance['geometry']} must contain an object"
        )
    provenance["geometry_sha256"] = hashlib.sha256(payload).hexdigest()
    return config, provenance


def check_arguments(args: argparse.Namespace) -> None:
    for name in ("fit_q_range", "diagnostic_range"):
        low, high = getattr(args, name)
        if not np.all(np.isfinite([low, high])) or not low < high:
            raise ValueError(f"{name} requires finite increasing bounds")
    if args.fit_q_range[0] <= 0:
        raise ValueError("Positive fit Q bounds required")
    for name in ("iterations", "overlap_min_bins", "jpeg_dpi"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in (
        "saturation",
        "density",
        "overlap_min_pixels",
        "overlap_floor",
        "q_step_size",
        "two_theta_step_size",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(args.angle_tolerance) or args.angle_tolerance < 0:
        raise ValueError("angle_tolerance must be finite and nonnegative")
    if args.q_max is not None and (not np.isfinite(args.q_max) or args.q_max <= 0):
        raise ValueError("q_max must be None or finite and positive (A^-1)")
    if not np.isfinite(args.polarization) or not -1 <= args.polarization <= 1:
        raise ValueError("polarization must be finite and in [-1, 1]")
    if not 16 <= radial_bins(tuple(args.fit_q_range), args.q_step_size) <= 65535:
        raise ValueError("Fit Q range and q_step_size must produce 16..65535 bins")


def radial_bins(limits: tuple[float, float], step: float) -> int:
    """Cover exact endpoints with uniform bins no wider than the target step."""
    width = limits[1] - limits[0]
    if not np.isfinite(width) or width <= 0 or not np.isfinite(step) or step <= 0:
        raise ValueError("Radial range and step must be finite and positive")
    count = width / step
    if not np.isfinite(count) or count > np.iinfo(np.int32).max:
        raise ValueError("Radial step requests too many bins")
    # Avoid an extra bin due solely to floating-point division at an integer.
    return max(1, int(np.ceil(np.nextafter(count, -np.inf))))


def output_theta_range(
    q_max: float | None, wavelength_m: float, full_range: tuple[float, float]
) -> tuple[float, float]:
    """Convert the output-only Q limit (A^-1) to 2theta degrees."""
    if q_max is None:
        return full_range
    sine = q_max * (wavelength_m * 1e10) / (4 * np.pi)
    if not np.isfinite(sine) or not 0 < sine <= 1:
        raise ValueError("q_max must be positive and no greater than 4*pi/wavelength")
    return 0.0, float(np.rad2deg(2 * np.arcsin(sine)))


def integrate_theta(
    scan: Scan,
    model: PhysicalModel,
    fit: Mapping[str, Any],
    base_mask: BoolArray,
    theta_range: tuple[float, float],
    npt: int,
    group_sizes: Sequence[int],
) -> Payload:
    """One grouped-data pass: integrate only 2theta and collect sampling counts."""
    import eiger_correction.eiger_workflow as workflow

    mask = base_mask | fit["bad_pixel_mask"] | scan.native_mask
    results = {}
    observations = np.zeros((len(scan.angles), npt), dtype=np.int64)
    edges = np.linspace(theta_range[0], theta_range[1], npt + 1)
    with h5py.File(scan.detector_path, "r") as source:
        frames = workflow.h5_dataset(source, scan.detector_dataset)
        for index, angle in enumerate(scan.angles):
            counts = np.asarray(frames[index], dtype=float)
            results[index] = workflow.integrate_frame(
                model,
                counts,
                angle,
                scan.i0[index],
                mask,
                fit["combined_divisor"],
                npt,
                theta_range,
                unit="2th_deg",
                saturation=np.inf,
            )
            integrator = model.geometry.get_ai(float(angle))
            bad = mask | workflow.azimuth_mask(model, angle, integrator)
            bad |= ~np.isfinite(counts) | (counts < 0)
            theta = np.rad2deg(integrator.center_array(scan.shape, unit="2th_rad"))
            valid = (
                ~bad & np.isfinite(theta) & (theta >= edges[0]) & (theta <= edges[-1])
            )
            observations[index] = np.histogram(theta[valid], bins=edges)[0]
            if observations[index].sum() != np.count_nonzero(valid):
                raise ValueError("Pixel-centre sampling count conservation failed")
    group = workflow.merge_results(results)
    group["per_frame"] = np.array(
        [
            [
                getattr(item, field)
                for field in (
                    "sum_signal",
                    "sum_normalization",
                    "sum_variance",
                    "count",
                )
            ]
            for item in results.values()
        ]
    )
    q = (
        4
        * np.pi
        * np.sin(np.deg2rad(group["axis"]) / 2)
        / (model.geometry.wavelength * 1e9)
    )
    group["within_fit_range"] = (q >= fit["q_range"][0]) & (q <= fit["q_range"][1])
    return {
        "groups": {"merged_2theta": group},
        "frame_gains": np.ones(len(scan.angles)),
        "mask": mask,
        "saturation": np.inf,
        "azimuth_range_deg": model.azimuth_range,
        "sampling": {
            "observations_by_position": observations,
            "group_sizes": np.asarray(group_sizes),
            "theta_edges": edges,
        },
    }


def check_conservation(
    unity: Mapping[str, Any],
    final: Mapping[str, Any],
    scale_fit: Mapping[str, Any],
) -> dict[str, Any]:
    group = final["groups"]["merged_2theta"]
    original = unity["groups"]["merged_2theta"]["per_frame"]
    scaled = group["per_frame"]
    np.testing.assert_array_equal(scaled[:, (0, 2, 3)], original[:, (0, 2, 3)])
    np.testing.assert_allclose(
        scaled[:, 1], original[:, 1] * final["frame_gains"][:, None]
    )
    np.testing.assert_allclose(scaled.sum(axis=0), group["accumulators"])
    np.testing.assert_allclose(np.mean(np.log(final["frame_gains"])), 0, atol=1e-12)
    if scale_fit["n_components"] != 1 or scale_fit["n_positions"] != len(
        final["frame_gains"]
    ):
        raise ValueError("All positions must be connected in the overlap graph")
    totals = group["accumulators"]
    supported = totals[1] > 0
    if not supported.any():
        raise ValueError("No supported final output bins")
    if not np.all(np.isfinite(totals)) or np.any(totals[1:] < 0):
        raise ValueError("Invalid final accumulators")
    np.testing.assert_allclose(
        group["intensity"][supported], totals[0, supported] / totals[1, supported]
    )
    np.testing.assert_allclose(
        group["sigma"][supported], np.sqrt(totals[2, supported]) / totals[1, supported]
    )
    if not np.all(np.isfinite(group["intensity"][supported])) or not np.all(
        np.diff(group["axis"]) > 0
    ):
        raise ValueError(
            "Final supported intensity must be finite and axis strictly increasing"
        )
    return {
        "status": "passed",
        "supported_bins": int(supported.sum()),
        "unsupported_bins": int((~supported).sum()),
    }


def integrate_output(
    scan: Scan,
    model: PhysicalModel,
    fit: Mapping[str, Any],
    base_mask: BoolArray,
    theta_range: tuple[float, float],
    theta_step: float,
    q_step: float,
    gains: NDArray[np.floating[Any]],
) -> Payload:
    """Integrate final grids with frozen corrections; never refit on cropped data."""
    import eiger_correction.eiger_workflow as workflow

    q_max_nm = (
        4
        * np.pi
        * np.sin(np.deg2rad(theta_range[1]) / 2)
        / (model.geometry.wavelength * 1e9)
    )
    unity = workflow.process_scan(
        scan,
        {
            "fit_settings": {"q_range": fit["q_range"]},
            "base_mask": base_mask,
            "bad_pixel_mask": fit["bad_pixel_mask"],
            "combined_divisor": fit["combined_divisor"],
        },
        model,
        q_range=(0.0, float(q_max_nm)),
        q_npt=radial_bins((0.0, float(q_max_nm / 10)), q_step),
        theta_npt=radial_bins(theta_range, theta_step),
        theta_range=theta_range,
        saturation=np.inf,
    )
    # Results are redundant once the merge and per-position accumulators exist.
    unity.pop("results", None)
    output = workflow.apply_position_scales(unity, gains)
    output["groups"]["merged_q"]["axis"] = output["groups"]["merged_q"]["axis"] / 10
    output["groups"]["merged_q"]["unit"] = "q_A^-1"
    return output


def save_xy(directory: Path, processed: Mapping[str, Any]) -> dict[str, str]:
    """Write supported bins of both independently integrated radial grids."""
    outputs = {}
    for name, suffix, label in (
        ("merged_2theta", "tth", "2theta_deg"),
        ("merged_q", "q", "q_A^-1"),
    ):
        group = processed["groups"][name]
        supported = (group["accumulators"][1] > 0) & np.isfinite(group["intensity"])
        if not supported.any():
            raise ValueError(f"No supported final output bins in {name}")
        xy = np.column_stack((group["axis"][supported], group["intensity"][supported]))
        filename = f"pattern_{suffix}.xy"
        with (directory / filename).open("x", encoding="ascii") as stream:
            np.savetxt(
                stream,
                xy,
                fmt="%.10g",
                header=f"{label} intensity; two_stage_overlap; measured I0 retained",
            )
        np.testing.assert_allclose(
            np.loadtxt(directory / filename, ndmin=2), xy, rtol=6e-10, atol=1e-12
        )
        outputs[f"xy_{suffix}"] = filename
    return outputs


def raw_audit(
    workflow: ModuleType,
    raw: Scan,
    scan: Scan,
    groups: Sequence[NDArray[np.integer[Any]]],
    model: PhysicalModel,
    fit: Mapping[str, Any],
    final: Mapping[str, Any],
    theta_range: tuple[float, float],
    npt: int,
    saturation: float,
) -> dict[str, Any]:
    index = int(np.argmax(scan.angles))
    counts = np.zeros(scan.shape)
    invalid = np.zeros(scan.shape, dtype=bool)
    with h5py.File(raw.detector_path, "r") as source:
        frames = workflow.h5_dataset(source, raw.detector_dataset)
        for member in groups[index]:
            frame = np.asarray(frames[int(member)], dtype=float)
            bad = ~np.isfinite(frame) | (frame < 0) | (frame >= saturation)
            invalid |= bad
            counts += np.where(bad, 0, frame)
    counts[invalid] = np.nan
    result = workflow.integrate_frame(
        model,
        counts,
        scan.angles[index],
        scan.i0[index],
        final["mask"],
        fit["combined_divisor"],
        npt,
        theta_range,
        unit="2th_deg",
        frame_gain=final["frame_gains"][index],
        saturation=np.inf,
    )
    for field, attr in enumerate(
        ("sum_signal", "sum_normalization", "sum_variance", "count")
    ):
        np.testing.assert_allclose(
            getattr(result, attr),
            final["groups"]["merged_2theta"]["per_frame"][index, field],
            rtol=3e-6,
            atol=1e-20,
        )
    return {"status": "passed", "position": index, "raw_frames": len(groups[index])}


def extract(args: argparse.Namespace, directory: Path, record: RunRecord) -> None:
    # Register compressed detector filters and import scientific libraries inside
    # the captured run, so library messages/warnings go to run.log as well.
    import hdf5plugin  # noqa: F401

    import eiger_correction.eiger_workflow as workflow
    from eiger_correction.eiger_diagnostics import save_diagnostics

    with record.stage("inputs"):
        check_arguments(args)
        # Scientific fitting and saved calibration settings retain nm^-1.
        # Convert once at the CLI boundary, without mutating recorded arguments.
        fit_q_range = (float(args.fit_q_range[0] * 10), float(args.fit_q_range[1] * 10))
        fit_bins = radial_bins(tuple(args.fit_q_range), args.q_step_size)
        geometry, geometry_provenance = load_geometry(args.scan, args.geometry)
        record.data["inputs"] = {"scan": args.scan.resolve(), **geometry_provenance}
        raw = workflow.load_scan(args.scan, args.detector, args.monitor, args.metadata)
        # Tiny optional metadata read, not another detector-data pass.
        with h5py.File(raw.path, "r") as source:
            current_path = "entry/instrument/synchrotron/current"
            if current_path in source:
                try:
                    raw.ring_current = np.asarray(
                        workflow.h5_dataset(source, current_path)[()], dtype=float
                    ).reshape(-1)
                except (ValueError, TypeError, OSError):
                    record.data["warnings"].append(
                        "Optional ring-current metadata could not be decoded; "
                        "overlay omitted."
                    )
        azimuth = workflow.validate_azimuth_range(
            None if args.all_azimuths else args.azimuth_range
        )
        packages: dict[str, str] = {}
        for name in (
            "numpy",
            "scipy",
            "h5py",
            "pyFAI",
            "xraylib",
            "matplotlib",
            "hdf5plugin",
        ):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = "unknown"
        record.data["software"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "source_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in (
                    "extract_eiger_corrections.py",
                    "eiger_workflow.py",
                    "eiger_diagnostics.py",
                    "eiger_report.py",
                )
            },
        }
        record.data["inputs"] = {
            "scan": raw.path,
            "detector": raw.detector_path,
            **geometry_provenance,
            "raw_frames": len(raw.angles),
            "azimuth_range_deg": azimuth,
        }

    with tempfile.TemporaryDirectory(dir=directory, prefix=".grouped_") as temporary:
        with record.stage("aggregation"):
            scan, groups = workflow.aggregate_equal_angle_frames(
                raw,
                Path(temporary) / "frames.h5",
                saturation=args.saturation,
                angle_tolerance=args.angle_tolerance,
            )
            sizes = [len(members) for members in groups]
            np.testing.assert_array_equal(
                np.sort(np.concatenate(groups)), np.arange(len(raw.angles))
            )
            np.testing.assert_allclose(
                scan.angles,
                [np.mean(raw.angles[members]) for members in groups],
                rtol=0,
                atol=0,
            )
            np.testing.assert_allclose(
                scan.i0, [raw.i0[members].sum() for members in groups], rtol=1e-14
            )
            record.data["aggregation"] = {
                "positions": len(groups),
                "group_sizes": sizes,
                "angle_spans_deg": [
                    float(np.ptp(raw.angles[members])) for members in groups
                ],
                "angles_deg": scan.angles,
                "summed_i0": scan.i0,
                "checks": (
                    "Complete frame partition; arithmetic mean motor angles; summed I0"
                ),
            }
        with record.stage("geometry"):
            model = workflow.PhysicalModel(
                geometry, scan, args.polarization, args.density, azimuth_range=azimuth
            )
            boundary, sensors = workflow.detector_layout(scan.shape)
            base_mask = scan.native_mask | boundary
            sensors[base_mask] = -1
            vertical = model.vertical_check()
            workflow.constant_pattern_check(model, base_mask, fit_q_range, fit_bins)
            record.data["geometry_checks"] = {
                "vertical": vertical,
                "constant_pattern": "passed",
                "fixed_mask_pixels": int(base_mask.sum()),
            }
        options: dict[str, Any] = {
            "npt": fit_bins,
            "iterations": args.iterations,
            "fit_frame_gains": True,
            "fractional_floor": 0.003,
            "reject_z": 5.0,
            "minimum_observations": 8,
            "shrink_weight": 300.0,
            "frame_prior_sigma": 0.01,
        }
        settings = {
            **options,
            "q_range": fit_q_range,
            "q_range_unit": "nm^-1",
            "q_step_size_angstrom_inverse": args.q_step_size,
            "two_theta_step_size_deg": args.two_theta_step_size,
            "azimuth_range_deg": azimuth,
            "angle_aggregation": {
                "enabled": True,
                "method": "sum",
                "angle_tolerance_deg": args.angle_tolerance,
                "source_frame_count": len(raw.angles),
                "analysis_position_count": len(groups),
                "group_sizes": sizes,
            },
        }
        with record.stage("fit_cache"):
            cache = workflow.build_cache(
                scan,
                model,
                base_mask,
                fit_q_range,
                fit_bins,
                saturation=np.inf,
            )
            indices = np.array(sorted(cache), dtype=int)
            record.data["cache"] = {
                "positions": indices,
                "observations": sum(len(item["pixels"]) for item in cache.values()),
                "bytes": sum(
                    array.nbytes for item in cache.values() for array in item.values()
                ),
            }
        reports: list[dict[str, Any]] = []
        with record.stage("held_out_validation"):
            if args.validation == "basic":
                record.data["warnings"].append(
                    "Independent detector validation disabled (--validation basic)."
                )
            else:
                folds = [(indices[::2], indices[1::2])]
                if args.validation == "full":
                    folds.append((indices[1::2], indices[::2]))
                sensor_gains: list[NDArray[np.floating[Any]]] = []
                for fold, (train, test) in enumerate(folds, start=1):
                    held_fit = workflow.fit_response(
                        cache,
                        train,
                        sensors,
                        len(scan.angles),
                        **{**options, "minimum_observations": 4},
                    )
                    validation = workflow.validate_response(
                        scan,
                        model,
                        cache,
                        held_fit,
                        test,
                        base_mask,
                        fit_q_range,
                        fit_bins,
                        saturation=np.inf,
                        validate_frame_gains=args.validation == "full",
                    )
                    reports.append(
                        {
                            "fold": fold,
                            "train_positions": train.tolist(),
                            "test_positions": test.tolist(),
                            "minimum_observations": 4,
                            "validate_frame_gains": args.validation == "full",
                            "metrics": validation["summary"],
                            "frame_gain_checks": validation["frame_rows"],
                        }
                    )
                    sensor_gains.append(held_fit["sensor_gains"])
                    metrics = {row["label"]: row for row in validation["summary"]}
                    for metric in (
                        "absolute_overlap_percent",
                        "centered_overlap_percent",
                        "higher_q_rms_percent",
                    ):
                        before, after = (
                            metrics["physical_only"][metric],
                            metrics["detector_only"][metric],
                        )
                        if (
                            not np.isfinite(before)
                            or not np.isfinite(after)
                            or after > before
                        ):
                            record.data["warnings"].append(
                                f"Fold {fold}: detector-only {metric} did not "
                                "improve or is nonfinite; inspect held-out metrics."
                            )
                    record.data["held_out_validation"] = reports
                    record.save()
                    del held_fit, validation
                if len(sensor_gains) == 2:
                    record.data["fold_sensor_log_gain_rms_percent"] = float(
                        100
                        * np.sqrt(
                            np.mean(
                                (np.log(sensor_gains[0]) - np.log(sensor_gains[1])) ** 2
                            )
                        )
                    )
        with record.stage("detector_fit"):
            fit = workflow.fit_response(
                cache, indices, sensors, len(scan.angles), **options
            )
            fit["q_range"] = fit_q_range
            record.data["detector_fit"] = {
                "sensor_gains": fit["sensor_gains"],
                "bad_pixels": int(fit["bad_pixel_mask"].sum()),
                "history": fit["history"],
                "stage1_gains": fit["frame_gains"],
            }
            del cache
        with record.stage("theta_integration"):
            theta_range = workflow.theta_range_for_scan(
                scan,
                model,
                base_mask | fit["bad_pixel_mask"],
            )
            theta_bins = radial_bins(theta_range, args.two_theta_step_size)
            unity = integrate_theta(
                scan, model, fit, base_mask, theta_range, theta_bins, sizes
            )
            record.data["theta_range_deg"] = theta_range
        with record.stage("overlap_scales"):
            scale_fit = workflow.refine_position_scales(
                unity["groups"]["merged_2theta"],
                minimum_pixel_weight=args.overlap_min_pixels,
                minimum_overlap_bins=args.overlap_min_bins,
                fractional_floor=args.overlap_floor,
            )
            final = workflow.apply_position_scales(unity, scale_fit["frame_gains"])
            record.data["position_scale_refinement"] = scale_fit
            record.data["integrity_checks"] = check_conservation(
                unity, final, scale_fit
            )
            if record.data["integrity_checks"]["unsupported_bins"]:
                record.data["warnings"].append(
                    "Unsupported final bins are omitted from XY; inspect HDF5 coverage."
                )
        if args.raw_audit:
            with record.stage("raw_audit"):
                record.data["raw_audit"] = raw_audit(
                    workflow,
                    raw,
                    scan,
                    groups,
                    model,
                    fit,
                    final,
                    theta_range,
                    theta_bins,
                    args.saturation,
                )
        else:
            record.data["raw_audit"] = {"status": "not_requested"}
        with record.stage("final_integration"):
            output_range = output_theta_range(
                args.q_max, model.geometry.wavelength, theta_range
            )
            output = integrate_output(
                scan,
                model,
                fit,
                base_mask,
                output_range,
                args.two_theta_step_size,
                args.q_step_size,
                final["frame_gains"],
            )
            record.data["output_theta_range_deg"] = output_range
        with record.stage("exports"):
            settings.update(
                position_scale_model="two_stage_overlap",
                refine_all_position_scales=True,
                position_scale_refinement={
                    **scale_fit["settings"],
                    "range": "full unmasked scan 2theta range; independent of output crop",
                    "theta_range_deg": theta_range,
                    "gauge": scale_fit["gauge"],
                    "replaces_stage1_gains": True,
                },
            )
            calibration_path = directory / "detector_corrections.h5"
            export_fit = {**fit, "frame_gains": final["frame_gains"]}
            workflow.save_calibration(
                calibration_path,
                scan,
                model,
                export_fit,
                base_mask,
                sensors,
                settings,
                reports,
            )
            # Augment only our newly-created local output; never source files.
            with h5py.File(calibration_path, "r+") as target:
                diagnostic = workflow.h5_group(target, "fit_diagnostics")
                nuisance = diagnostic.create_dataset(
                    "detector_fit_nuisance_frame_gains", data=fit["frame_gains"]
                )
                nuisance.attrs["stage"] = (
                    "stage 1 diagnostic only; not the final applied gains"
                )
                diagnostic.attrs["validation_scope"] = record.data["validation_scope"]
                diagnostic.attrs["frame_gains_stage"] = "full-range overlaps"
                diagnostic.attrs["position_scale_refinement_json"] = json.dumps(
                    json_safe(scale_fit), allow_nan=False
                )
                diagnostic.attrs["scale_uncertainty"] = (
                    "not propagated; overlap edges and split bins are correlated"
                )
            calibration = workflow.load_calibration(calibration_path)
            np.testing.assert_array_equal(
                calibration["frame_gains"], final["frame_gains"]
            )
            workflow.save_processed(
                directory / "pattern.h5", scan, calibration, model, output
            )
            record.data["outputs"] = {
                "calibration": calibration_path.name,
                "pattern": "pattern.h5",
                **save_xy(directory, output),
                "log": "run.log",
            }
        with record.stage("jpeg_diagnostics"):
            summary = save_diagnostics(
                directory,
                raw,
                scan,
                groups,
                model,
                base_mask,
                fit,
                unity,
                final,
                scale_fit,
                theta_range=theta_range,
                diagnostic_range=tuple(args.diagnostic_range),
                min_pixel_weight=args.overlap_min_pixels,
                dpi=args.jpeg_dpi,
            )
            record.data["diagnostics"] = summary
            record.data["warnings"].extend(summary.get("warnings", []))
