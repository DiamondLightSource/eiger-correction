"""Headless, two-stage EIGER scan extraction with durable diagnostics and reporting."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback
import warnings

import h5py
import numpy as np

from eiger_report import write_markdown_report


def json_safe(value):
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

    def __init__(self, directory, arguments):
        self.path = directory / "report.json"
        self.started = time.perf_counter()
        self.data = {
            "schema": "eiger_extraction_report_v1", "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(arguments), "stages": {}, "warnings": [],
            "validation_scope": "Held-out checks test stage 1 only. Stage-2 overlap residuals are in-sample.",
            "limitations": [
                "Detector response outside the fit Q range is extrapolated.",
                "Poisson sigma excludes scale, calibration, I0 and model uncertainty and bin covariance.",
                "Source-position gains are source-scan-only, not transferable corrections.",
                "Successful integrity checks do not certify scientific accuracy.",
            ],
        }
        self.save()

    def save(self):
        self.data["elapsed_seconds"] = time.perf_counter() - self.started
        snapshot = json_safe(self.data)
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(snapshot, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(self.path)
        write_markdown_report(self.path.with_suffix(".md"), snapshot)

    @contextmanager
    def stage(self, name):
        entry = self.data["stages"][name] = {"status": "running"}
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


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("scan", type=Path, help="Source NeXus scan (read-only)")
    result.add_argument("--geometry", type=Path,
                        help="Override goniometer calibration JSON; default: /entry/plan_metadata/geometry_calibration in the scan")
    result.add_argument("--output-dir", type=Path, help="New directory; existing directories are never overwritten")
    for name in ("detector", "monitor", "metadata"):
        result.add_argument(f"--{name}", type=Path, help=f"Override companion {name} file")
    result.add_argument("--azimuth-range", type=float, nargs=2, default=(67.5, 112.5), metavar=("LOW", "HIGH"))
    result.add_argument("--all-azimuths", action="store_true", help="Disable the default single chi wedge")
    result.add_argument("--fit-q-range", type=float, nargs=2, default=(100., 400.), metavar=("LOW", "HIGH"), help="nm^-1")
    result.add_argument("--fit-bins", type=int, default=1200)
    result.add_argument("--iterations", type=int, default=10)
    result.add_argument("--theta-bins", type=int, default=2001)
    result.add_argument("--theta-max", type=float, default=80., help="Upper 2theta in degrees; 0 selects automatic unmasked pixel-centre maximum")
    result.add_argument("--angle-tolerance", type=float, default=0.001)
    result.add_argument("--saturation", type=float, default=1e7, help="Raw-frame rejection threshold (not applied to valid sums)")
    result.add_argument("--polarization", type=float, default=0.95)
    result.add_argument("--density", type=float, default=6.2, help="CdTe density in g/cm^3")
    result.add_argument("--overlap-min-pixels", type=float, default=100.)
    result.add_argument("--overlap-min-bins", type=int, default=20)
    result.add_argument("--overlap-floor", type=float, default=0.003)
    result.add_argument("--validation", choices=("basic", "holdout", "full"), default="holdout",
                        help="Always integrity checks; holdout adds one independent detector fold, full adds both folds and lower-Q gain checks")
    result.add_argument("--raw-audit", action="store_true", help="Reread and reintegrate the highest-position raw group independently")
    result.add_argument("--diagnostic-range", type=float, nargs=2, default=(30., 80.), metavar=("LOW", "HIGH"))
    result.add_argument("--jpeg-dpi", type=int, default=140)
    return result


def load_geometry(scan_path, geometry_path=None):
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
        message = f"No usable embedded geometry JSON at {path}::{dataset_path}; supply --geometry JSON_FILE"
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
        provenance = {"geometry": f"{path}::{dataset_path}", "geometry_source": "nexus",
                      "geometry_dataset": dataset_path}
    try:
        config = json.loads(payload)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid geometry JSON from {provenance['geometry']}; supply a valid --geometry JSON_FILE") from error
    if not isinstance(config, dict):
        raise ValueError(f"Geometry JSON from {provenance['geometry']} must contain an object")
    provenance["geometry_sha256"] = hashlib.sha256(payload).hexdigest()
    return config, provenance


def check_arguments(args):
    for name in ("fit_q_range", "diagnostic_range"):
        low, high = getattr(args, name)
        if not np.all(np.isfinite([low, high])) or not low < high:
            raise ValueError(f"{name} requires finite increasing bounds")
    if args.fit_q_range[0] <= 0 or not 16 <= args.fit_bins <= 65535:
        raise ValueError("Positive fit Q bounds and 16..65535 fit bins required")
    for name in ("iterations", "theta_bins", "overlap_min_bins", "jpeg_dpi"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("saturation", "density", "overlap_min_pixels", "overlap_floor"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(args.angle_tolerance) or args.angle_tolerance < 0:
        raise ValueError("angle_tolerance must be finite and nonnegative")
    if not np.isfinite(args.theta_max) or not 0 <= args.theta_max <= 180:
        raise ValueError("theta_max must be 0 (automatic) or in (0, 180]")
    if not np.isfinite(args.polarization) or not -1 <= args.polarization <= 1:
        raise ValueError("polarization must be finite and in [-1, 1]")


def integrate_theta(scan, model, fit, base_mask, theta_range, npt, group_sizes):
    """One grouped-data pass: integrate only 2theta and collect sampling counts."""
    import eiger_workflow as workflow

    mask = base_mask | fit["bad_pixel_mask"] | scan.native_mask
    results = {}
    observations = np.zeros((len(scan.angles), npt), dtype=np.int64)
    edges = np.linspace(*theta_range, npt + 1)
    with h5py.File(scan.detector_path, "r") as source:
        for index, angle in enumerate(scan.angles):
            counts = np.asarray(source[scan.detector_dataset][index], dtype=float)
            results[index] = workflow.integrate_frame(
                model, counts, angle, scan.i0[index], mask, fit["combined_divisor"],
                npt, theta_range, unit="2th_deg", saturation=np.inf,
            )
            integrator = model.geometry.get_ai(float(angle))
            bad = mask | workflow.azimuth_mask(model, angle, integrator)
            bad |= ~np.isfinite(counts) | (counts < 0)
            theta = np.rad2deg(integrator.center_array(scan.shape, unit="2th_rad"))
            valid = ~bad & np.isfinite(theta) & (theta >= edges[0]) & (theta <= edges[-1])
            observations[index] = np.histogram(theta[valid], bins=edges)[0]
            if observations[index].sum() != np.count_nonzero(valid):
                raise ValueError("Pixel-centre sampling count conservation failed")
    group = workflow.merge_results(results)
    group["per_frame"] = np.array([
        [getattr(item, field) for field in ("sum_signal", "sum_normalization", "sum_variance", "count")]
        for item in results.values()
    ])
    q = 4 * np.pi * np.sin(np.deg2rad(group["axis"]) / 2) / (model.geometry.wavelength * 1e9)
    group["within_fit_range"] = (q >= fit["q_range"][0]) & (q <= fit["q_range"][1])
    return {
        "groups": {"merged_2theta": group}, "frame_gains": np.ones(len(scan.angles)),
        "mask": mask, "saturation": np.inf, "azimuth_range_deg": model.azimuth_range,
        "sampling": {"observations_by_position": observations, "group_sizes": np.asarray(group_sizes), "theta_edges": edges},
    }


def check_conservation(unity, final, scale_fit):
    group = final["groups"]["merged_2theta"]
    original = unity["groups"]["merged_2theta"]["per_frame"]
    scaled = group["per_frame"]
    np.testing.assert_array_equal(scaled[:, (0, 2, 3)], original[:, (0, 2, 3)])
    np.testing.assert_allclose(scaled[:, 1], original[:, 1] * final["frame_gains"][:, None])
    np.testing.assert_allclose(scaled.sum(axis=0), group["accumulators"])
    np.testing.assert_allclose(np.mean(np.log(final["frame_gains"])), 0, atol=1e-12)
    if scale_fit["n_components"] != 1 or scale_fit["n_positions"] != len(final["frame_gains"]):
        raise ValueError("All positions must be connected in the overlap graph")
    totals = group["accumulators"]
    supported = totals[1] > 0
    if not supported.any():
        raise ValueError("No supported final output bins")
    if not np.all(np.isfinite(totals)) or np.any(totals[1:] < 0):
        raise ValueError("Invalid final accumulators")
    np.testing.assert_allclose(group["intensity"][supported], totals[0, supported] / totals[1, supported])
    np.testing.assert_allclose(group["sigma"][supported], np.sqrt(totals[2, supported]) / totals[1, supported])
    if not np.all(np.isfinite(group["intensity"][supported])) or not np.all(np.diff(group["axis"]) > 0):
        raise ValueError("Final supported intensity must be finite and axis strictly increasing")
    return {"status": "passed", "supported_bins": int(supported.sum()), "unsupported_bins": int((~supported).sum())}


def raw_audit(workflow, raw, scan, groups, model, fit, final, theta_range, npt, saturation):
    index = int(np.argmax(scan.angles))
    counts = np.zeros(scan.shape)
    invalid = np.zeros(scan.shape, dtype=bool)
    with h5py.File(raw.detector_path, "r") as source:
        for member in groups[index]:
            frame = np.asarray(source[raw.detector_dataset][int(member)], dtype=float)
            bad = ~np.isfinite(frame) | (frame < 0) | (frame >= saturation)
            invalid |= bad
            counts += np.where(bad, 0, frame)
    counts[invalid] = np.nan
    result = workflow.integrate_frame(
        model, counts, scan.angles[index], scan.i0[index], final["mask"], fit["combined_divisor"],
        npt, theta_range, unit="2th_deg", frame_gain=final["frame_gains"][index], saturation=np.inf,
    )
    for field, attr in enumerate(("sum_signal", "sum_normalization", "sum_variance", "count")):
        np.testing.assert_allclose(getattr(result, attr), final["groups"]["merged_2theta"]["per_frame"][index, field], rtol=3e-6, atol=1e-20)
    return {"status": "passed", "position": index, "raw_frames": len(groups[index])}


def extract(args, directory, record):
    # Register compressed detector filters and import scientific libraries inside
    # the captured run, so library messages/warnings go to run.log as well.
    import hdf5plugin  # noqa: F401
    import eiger_workflow as workflow
    from eiger_diagnostics import save_diagnostics

    with record.stage("inputs"):
        check_arguments(args)
        geometry, geometry_provenance = load_geometry(args.scan, args.geometry)
        record.data["inputs"] = {"scan": args.scan.resolve(), **geometry_provenance}
        raw = workflow.load_scan(args.scan, args.detector, args.monitor, args.metadata)
        # Tiny optional metadata read, not another detector-data pass.
        with h5py.File(raw.path, "r") as source:
            current_path = "entry/instrument/synchrotron/current"
            if current_path in source:
                try:
                    raw.ring_current = np.asarray(source[current_path][()], dtype=float).reshape(-1)
                except (ValueError, TypeError, OSError):
                    record.data["warnings"].append("Optional ring-current metadata could not be decoded; overlay omitted.")
        azimuth = workflow.validate_azimuth_range(None if args.all_azimuths else args.azimuth_range)
        packages = {}
        for name in ("numpy", "scipy", "h5py", "pyFAI", "xraylib", "matplotlib", "hdf5plugin"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = "unknown"
        record.data["software"] = {"python": sys.version, "platform": platform.platform(), "packages": packages,
                                   "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                     for name in ("extract_eiger_corrections.py", "eiger_workflow.py", "eiger_diagnostics.py", "eiger_report.py")}}
        record.data["inputs"] = {"scan": raw.path, "detector": raw.detector_path,
                                 **geometry_provenance,
                                 "raw_frames": len(raw.angles), "azimuth_range_deg": azimuth}

    with tempfile.TemporaryDirectory(dir=directory, prefix=".grouped_") as temporary:
        with record.stage("aggregation"):
            scan, groups = workflow.aggregate_equal_angle_frames(
                raw, Path(temporary) / "frames.h5", saturation=args.saturation, angle_tolerance=args.angle_tolerance,
            )
            sizes = [len(members) for members in groups]
            np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(len(raw.angles)))
            np.testing.assert_allclose(scan.angles, [np.mean(raw.angles[members]) for members in groups], rtol=0, atol=0)
            np.testing.assert_allclose(scan.i0, [raw.i0[members].sum() for members in groups], rtol=1e-14)
            record.data["aggregation"] = {"positions": len(groups), "group_sizes": sizes,
                                          "angle_spans_deg": [float(np.ptp(raw.angles[members])) for members in groups],
                                          "angles_deg": scan.angles, "summed_i0": scan.i0,
                                          "checks": "Complete frame partition; arithmetic mean motor angles; summed I0"}
        with record.stage("geometry"):
            model = workflow.PhysicalModel(geometry, scan, args.polarization, args.density, azimuth_range=azimuth)
            boundary, sensors = workflow.detector_layout(scan.shape)
            base_mask = scan.native_mask | boundary
            sensors[base_mask] = -1
            vertical = model.vertical_check()
            workflow.constant_pattern_check(model, base_mask, args.fit_q_range, args.fit_bins)
            record.data["geometry_checks"] = {"vertical": vertical, "constant_pattern": "passed", "fixed_mask_pixels": int(base_mask.sum())}
        options = {"npt": args.fit_bins, "iterations": args.iterations, "fit_frame_gains": True,
                   "fractional_floor": 0.003, "reject_z": 5., "minimum_observations": 8,
                   "shrink_weight": 300., "frame_prior_sigma": 0.01}
        settings = {**options, "q_range": args.fit_q_range, "azimuth_range_deg": azimuth,
                    "angle_aggregation": {"enabled": True, "method": "sum", "angle_tolerance_deg": args.angle_tolerance,
                                          "source_frame_count": len(raw.angles), "analysis_position_count": len(groups), "group_sizes": sizes}}
        with record.stage("fit_cache"):
            cache = workflow.build_cache(scan, model, base_mask, args.fit_q_range, args.fit_bins, saturation=np.inf)
            indices = np.array(sorted(cache), dtype=int)
            record.data["cache"] = {"positions": indices, "observations": sum(len(item["pixels"]) for item in cache.values()),
                                    "bytes": sum(array.nbytes for item in cache.values() for array in item.values())}
        reports = []
        with record.stage("held_out_validation"):
            if args.validation == "basic":
                record.data["warnings"].append("Independent detector validation disabled (--validation basic).")
            else:
                folds = [(indices[::2], indices[1::2])]
                if args.validation == "full":
                    folds.append((indices[1::2], indices[::2]))
                sensor_gains = []
                for fold, (train, test) in enumerate(folds, start=1):
                    held_fit = workflow.fit_response(cache, train, sensors, len(scan.angles), **{**options, "minimum_observations": 4})
                    validation = workflow.validate_response(
                        scan, model, cache, held_fit, test, base_mask, args.fit_q_range, args.fit_bins,
                        saturation=np.inf, validate_frame_gains=args.validation == "full",
                    )
                    reports.append({"fold": fold, "train_positions": train.tolist(), "test_positions": test.tolist(),
                                    "minimum_observations": 4, "validate_frame_gains": args.validation == "full",
                                    "metrics": validation["summary"], "frame_gain_checks": validation["frame_rows"]})
                    sensor_gains.append(held_fit["sensor_gains"])
                    metrics = {row["label"]: row for row in validation["summary"]}
                    for metric in ("absolute_overlap_percent", "centered_overlap_percent", "higher_q_rms_percent"):
                        before, after = metrics["physical_only"][metric], metrics["detector_only"][metric]
                        if not np.isfinite(before) or not np.isfinite(after) or after > before:
                            record.data["warnings"].append(f"Fold {fold}: detector-only {metric} did not improve or is nonfinite; inspect held-out metrics.")
                    record.data["held_out_validation"] = reports
                    record.save()
                    del held_fit, validation
                if len(sensor_gains) == 2:
                    record.data["fold_sensor_log_gain_rms_percent"] = float(100 * np.sqrt(np.mean((np.log(sensor_gains[0]) - np.log(sensor_gains[1])) ** 2)))
        with record.stage("detector_fit"):
            fit = workflow.fit_response(cache, indices, sensors, len(scan.angles), **options)
            fit["q_range"] = args.fit_q_range
            record.data["detector_fit"] = {"sensor_gains": fit["sensor_gains"], "bad_pixels": int(fit["bad_pixel_mask"].sum()),
                                           "history": fit["history"], "stage1_gains": fit["frame_gains"]}
            del cache
        with record.stage("theta_integration"):
            theta_range = workflow.theta_range_for_scan(scan, model, base_mask | fit["bad_pixel_mask"], theta_max=args.theta_max or None)
            unity = integrate_theta(scan, model, fit, base_mask, theta_range, args.theta_bins, sizes)
            record.data["theta_range_deg"] = theta_range
        with record.stage("overlap_scales"):
            scale_fit = workflow.refine_position_scales(
                unity["groups"]["merged_2theta"], minimum_pixel_weight=args.overlap_min_pixels,
                minimum_overlap_bins=args.overlap_min_bins, fractional_floor=args.overlap_floor,
            )
            final = workflow.apply_position_scales(unity, scale_fit["frame_gains"])
            record.data["position_scale_refinement"] = scale_fit
            record.data["integrity_checks"] = check_conservation(unity, final, scale_fit)
            if record.data["integrity_checks"]["unsupported_bins"]:
                record.data["warnings"].append("Unsupported final bins are omitted from XY; inspect HDF5 coverage.")
        if args.raw_audit:
            with record.stage("raw_audit"):
                record.data["raw_audit"] = raw_audit(workflow, raw, scan, groups, model, fit, final, theta_range, args.theta_bins, args.saturation)
        else:
            record.data["raw_audit"] = {"status": "not_requested"}
        with record.stage("exports"):
            settings.update(position_scale_model="two_stage_overlap", refine_all_position_scales=True,
                            position_scale_refinement={**scale_fit["settings"], "range": "full exported 2theta range",
                                                       "theta_range_deg": theta_range, "gauge": scale_fit["gauge"], "replaces_stage1_gains": True})
            calibration_path = directory / "detector_corrections.h5"
            export_fit = {**fit, "frame_gains": final["frame_gains"]}
            workflow.save_calibration(calibration_path, scan, model, export_fit, base_mask, sensors, settings, reports)
            # Augment only our newly-created local output; never source files.
            with h5py.File(calibration_path, "r+") as target:
                diagnostic = target["fit_diagnostics"]
                nuisance = diagnostic.create_dataset("detector_fit_nuisance_frame_gains", data=fit["frame_gains"])
                nuisance.attrs["stage"] = "stage 1 diagnostic only; not the final applied gains"
                diagnostic.attrs["validation_scope"] = record.data["validation_scope"]
                diagnostic.attrs["frame_gains_stage"] = "full-range overlaps"
                diagnostic.attrs["position_scale_refinement_json"] = json.dumps(json_safe(scale_fit), allow_nan=False)
                diagnostic.attrs["scale_uncertainty"] = "not propagated; overlap edges and split bins are correlated"
            calibration = workflow.load_calibration(calibration_path)
            np.testing.assert_array_equal(calibration["frame_gains"], final["frame_gains"])
            workflow.save_processed(directory / "pattern.h5", scan, calibration, model, final)
            group = final["groups"]["merged_2theta"]
            supported = (group["accumulators"][1] > 0) & np.isfinite(group["intensity"])
            xy = np.column_stack((group["axis"][supported], group["intensity"][supported]))
            with (directory / "pattern_2theta.xy").open("x", encoding="ascii") as stream:
                np.savetxt(stream, xy, fmt="%.10g", header="2theta_deg intensity; two_stage_overlap; measured I0 retained")
            np.testing.assert_allclose(np.loadtxt(directory / "pattern_2theta.xy", ndmin=2), xy, rtol=6e-10, atol=1e-12)
            record.data["outputs"] = {"calibration": calibration_path.name, "pattern": "pattern.h5", "xy": "pattern_2theta.xy", "log": "run.log"}
        with record.stage("jpeg_diagnostics"):
            diagnostic = save_diagnostics(
                directory, raw, scan, groups, model, base_mask, fit, unity, final, scale_fit,
                theta_range=theta_range, diagnostic_range=tuple(args.diagnostic_range),
                min_pixel_weight=args.overlap_min_pixels, dpi=args.jpeg_dpi,
            )
            record.data["diagnostics"] = diagnostic
            record.data["warnings"].extend(diagnostic.get("warnings", []))


def main(argv=None):
    args = parser().parse_args(argv)
    directory = (args.output_dir or Path("outputs") / f"{args.scan.stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}").resolve()
    # Exclusive directory creation prevents interleaved jobs and output clobbering.
    directory.mkdir(parents=True, exist_ok=False)
    record = RunRecord(directory, args)
    with (directory / "run.log").open("x", encoding="utf-8", buffering=1) as log:
        handler = logging.StreamHandler(log)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        logging.captureWarnings(True)
        try:
            with redirect_stdout(log), redirect_stderr(log), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    extract(args, directory, record)
                    record.data["status"] = "completed"
                except BaseException as error:
                    record.data.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                       error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
                    traceback.print_exc(file=log)
                finally:
                    record.data["warnings"].extend(f"{item.category.__name__}: {item.message}" for item in caught)
                    if record.data["status"] == "completed" and record.data["warnings"]:
                        record.data["status"] = "completed_with_warnings"
                    record.data["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    record.save()
        finally:
            logging.captureWarnings(False)
            root.removeHandler(handler)
    return 0 if record.data["status"].startswith("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())