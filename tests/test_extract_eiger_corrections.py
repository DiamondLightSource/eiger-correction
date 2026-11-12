"""Synthetic CLI, accounting and headless diagnostic regression tests.

No beamline data are used. TemporaryDirectory owns only test-created files;
scientific fitting/geometry are mocked, while array accounting and exporters
remain real wherever their public contracts permit it.
"""

import copy
import io
import json
import logging
import sys
import tempfile
import unittest
import warnings
from contextlib import ExitStack, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

import eiger_correction.__main__ as cli_main
import eiger_correction.eiger_diagnostics as diagnostics
import eiger_correction.eiger_workflow as workflow
import eiger_correction.extract_eiger_corrections as cli
from eiger_correction.eiger_workflow import h5_dataset, h5_group


def stub(**attributes: Any) -> Any:
    """Stand-in for Scan or PhysicalModel."""
    return SimpleNamespace(**attributes)


def strict_json(path):
    def reject_constant(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")

    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def processed_payload(n_positions=4, bins=8, shape=(4, 6)):
    """Positive, overlapping accumulators with nonidentical position curves."""
    axis = np.linspace(32.0, 76.0, bins)
    normalization = (1 + np.arange(n_positions)[:, None] / 10) * np.ones(
        (n_positions, bins)
    )
    signal = (
        normalization
        * (10 + np.arange(bins)[None, :])
        * np.linspace(0.97, 1.03, n_positions)[:, None]
    )
    frames = np.stack(
        (signal, normalization, signal + 2, np.full_like(signal, 150.0)), axis=1
    )
    totals = frames.sum(axis=0)
    group = {
        "axis": axis,
        "per_frame": frames,
        "accumulators": totals,
        "intensity": totals[0] / totals[1],
        "sigma": np.sqrt(totals[2]) / totals[1],
        "within_fit_range": np.arange(bins) < bins // 2,
    }
    return {
        "groups": {"merged_2theta": group},
        "frame_gains": np.ones(n_positions),
        "mask": np.zeros(shape, dtype=bool),
        "saturation": np.inf,
        "azimuth_range_deg": (67.5, 112.5),
        "sampling": {
            "observations_by_position": np.full((n_positions, bins), 3, dtype=np.int64),
            "group_sizes": np.full(n_positions, 2, dtype=np.int64),
            "theta_edges": np.linspace(30.0, 80.0, bins + 1),
        },
    }


def response_fit(shape=(4, 6), n_positions=4, bins=16):
    unity = np.ones(shape)
    return {
        "pixel_divisor": unity.copy(),
        "sensor_divisor": unity.copy(),
        "combined_divisor": unity.copy(),
        "sensor_gains": np.array([0.99, 1.01]),
        "bad_pixel_mask": np.zeros(shape, dtype=bool),
        "observations": np.full(shape, 12, dtype=np.int32),
        "outlier_fraction": np.zeros(shape),
        "frame_gains": np.linspace(0.98, 1.02, n_positions),
        "history": [],
        "fit_indices": np.arange(n_positions),
        "profile_log": np.zeros(bins),
        "supported_q_bins": np.ones(bins, dtype=bool),
        "q_range": (100.0, 400.0),
    }


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="eiger_cli_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def argv(self, output=None, *options):
        return [
            str(self.root / "scan.nxs"),
            "--geometry",
            str(self.root / "geometry.json"),
            "--output-dir",
            str(output or self.root / "run"),
            *options,
        ]


class ArgumentTests(TemporaryCase):
    def test_radial_bins_cover_endpoints_at_target_resolution(self):
        self.assertEqual(cli.radial_bins((10.0, 40.0), 0.025), 1200)
        self.assertEqual(cli.radial_bins((0.0, 80.0), 0.025), 3200)
        self.assertEqual(cli.radial_bins((0.0, 1.01), 0.025), 41)
        self.assertEqual(cli.radial_bins((0.0, 0.001), 0.025), 1)
        for limits, step in (((1, 0), 1), ((0, 1), 0), ((0, np.inf), 1)):
            with self.subTest(limits=limits, step=step), self.assertRaises(ValueError):
                cli.radial_bins(limits, step)

    def test_defaults_and_supported_boundary_values(self):
        args = cli.parser().parse_args(self.argv())
        cli.check_arguments(args)
        self.assertIsInstance(args.scan, Path)
        self.assertEqual(args.validation, "holdout")
        self.assertEqual(args.fit_q_range, (10.0, 40.0))
        self.assertEqual(args.q_step_size, 0.025)
        self.assertEqual(args.two_theta_step_size, 0.025)
        self.assertIsNone(args.q_max)
        for values in (
            {"q_max": None, "angle_tolerance": 0.0, "polarization": -1.0},
            {"q_max": 40.0, "polarization": 1.0},
        ):
            with self.subTest(values=values):
                cli.check_arguments(stub(**{**vars(args), **values}))

    def test_parser_rejects_missing_or_malformed_arguments(self):
        cases = [
            [],
            ["scan", "--geometry"],
            self.argv(None, "--validation", "unknown"),
            self.argv(None, "--q-step-size", "abc"),
            self.argv(None, "--fit-bins", "100"),
            self.argv(None, "--theta-bins", "100"),
            self.argv(None, "--theta-max", "80"),
            self.argv(None, "--azimuth-range", "20"),
            self.argv(None, "--unknown-option"),
        ]
        for argv in cases:
            with (
                self.subTest(argv=argv),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                cli.parser().parse_args(argv)
            self.assertEqual(caught.exception.code, 2)

    def test_geometry_is_optional(self):
        args = cli.parser().parse_args(["scan.nxs"])
        self.assertIsNone(args.geometry)
        cli.check_arguments(args)

    def test_check_arguments_rejects_invalid_values(self):
        args = cli.parser().parse_args(self.argv())
        cases = {
            "fit_q_range": [(1, 1), (2, 1), (0, 2), (-1, 2), (1, np.inf), (np.nan, 2)],
            "diagnostic_range": [(2, 1), (1, 1), (1, np.inf), (np.nan, 2)],
            "q_step_size": [0, -1, np.inf, np.nan, 10, 0.00001],
            "two_theta_step_size": [0, -1, np.inf, np.nan],
            "iterations": [0, -1],
            "overlap_min_bins": [0],
            "jpeg_dpi": [0],
            "saturation": [0, -1, np.inf, np.nan],
            "density": [0, -1, np.inf, np.nan],
            "overlap_min_pixels": [0, -1, np.inf, np.nan],
            "overlap_floor": [0, -1, np.inf, np.nan],
            "angle_tolerance": [-1, np.inf, np.nan],
            "q_max": [0, -1, np.inf, np.nan],
            "polarization": [-1.01, 1.01, np.inf, np.nan],
        }
        for field, values in cases.items():
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    cli.check_arguments(stub(**{**vars(args), field: value}))

    def test_azimuth_invalids_use_workflow_validation(self):
        for wedge in ((0, 0), (0, 361), (0, -361), (np.nan, 20), (10, np.inf), (10,)):
            with self.subTest(wedge=wedge), self.assertRaises(ValueError):
                workflow.validate_azimuth_range(wedge)
        # Directed wedges may cross the chi seam or use unwrapped angles.
        for wedge in ((170, -170), (20, 10), (-181, 30), (10, 181)):
            with self.subTest(wedge=wedge):
                self.assertEqual(workflow.validate_azimuth_range(wedge), wedge)


class LifecycleTests(TemporaryCase):
    def assert_restored(self, stdout, stderr, handlers, warning_hook):
        self.assertIs(sys.stdout, stdout)
        self.assertIs(sys.stderr, stderr)
        self.assertEqual(logging.getLogger().handlers, handlers)
        self.assertIs(warnings.showwarning, warning_hook)

    def test_success_and_warning_reports_capture_logs_and_restore_state(self):
        for emit_warning in (False, True):
            directory = self.root / str(emit_warning)
            stdout, stderr = sys.stdout, sys.stderr
            handlers = list(logging.getLogger().handlers)
            warning_hook = warnings.showwarning

            def extraction(args, output, record, emit_warning=emit_warning):
                with record.stage("synthetic"):
                    print("stdout marker")
                    print("stderr marker", file=sys.stderr)
                    logging.getLogger("eiger-test").warning("logging marker")
                    record.data["metrics"] = {
                        "array": np.array([np.nan, np.inf, -np.inf, 2.0]),
                        "scalar": np.int64(7),
                        "path": output,
                    }
                    if emit_warning:
                        warnings.warn("synthetic warning", RuntimeWarning, stacklevel=2)

            with patch.object(cli_main, "extract", side_effect=extraction):
                self.assertEqual(cli_main.main(self.argv(directory)), 0)
            self.assert_restored(stdout, stderr, handlers, warning_hook)
            report = strict_json(directory / "report.json")
            self.assertEqual(
                report["status"],
                "completed_with_warnings" if emit_warning else "completed",
            )
            self.assertEqual(report["stages"]["synthetic"]["status"], "passed")
            self.assertEqual(report["metrics"]["array"], [None, None, None, 2.0])
            self.assertEqual(report["metrics"]["scalar"], 7)
            self.assertEqual(report["metrics"]["path"], str(directory))
            self.assertIn("finished_utc", report)
            self.assertGreaterEqual(report["elapsed_seconds"], 0)
            if emit_warning:
                self.assertTrue(
                    any(
                        "RuntimeWarning: synthetic warning" in value
                        for value in report["warnings"]
                    )
                )
            log = (directory / "run.log").read_text()
            for marker in ("stdout marker", "stderr marker", "logging marker"):
                self.assertIn(marker, log)
            self.assertFalse((directory / "report.json.tmp").exists())
            markdown = (directory / "report.md").read_text()
            self.assertIn(
                "COMPLETED WITH WARNINGS" if emit_warning else "COMPLETED", markdown
            )
            self.assertIn("[Machine-readable record](report.json)", markdown)
            if emit_warning:
                self.assertIn("synthetic warning", markdown)
            self.assertFalse((directory / "report.md.tmp").exists())

    def test_failure_and_interruption_checkpoint_and_cleanup(self):
        for error, status in (
            (ValueError("synthetic failure"), "failed"),
            (KeyboardInterrupt("synthetic interrupt"), "interrupted"),
        ):
            directory = self.root / status
            stdout, stderr = sys.stdout, sys.stderr
            handlers = list(logging.getLogger().handlers)
            warning_hook = warnings.showwarning

            def extraction(args, output, record, error=error):
                with tempfile.TemporaryDirectory(
                    dir=output, prefix=".grouped_"
                ) as temporary:
                    Path(temporary, "frames.h5").touch()
                    with record.stage("integration"):
                        warnings.warn(
                            "warning before failure", UserWarning, stacklevel=2
                        )
                        raise error

            with patch.object(cli_main, "extract", side_effect=extraction):
                self.assertEqual(cli_main.main(self.argv(directory)), 1)
            self.assert_restored(stdout, stderr, handlers, warning_hook)
            report = strict_json(directory / "report.json")
            self.assertEqual(report["status"], status)
            self.assertEqual(report["stages"]["integration"]["status"], "failed")
            self.assertGreaterEqual(report["stages"]["integration"]["seconds"], 0)
            self.assertEqual(report["error"]["type"], type(error).__name__)
            self.assertIn(str(error), report["error"]["traceback"])
            self.assertIn(type(error).__name__, (directory / "run.log").read_text())
            self.assertTrue(
                any("warning before failure" in item for item in report["warnings"])
            )
            self.assertFalse(list(directory.glob(".grouped_*")))
            self.assertFalse((directory / "report.json.tmp").exists())
            markdown = (directory / "report.md").read_text()
            self.assertIn(status.upper(), markdown)
            self.assertIn("Do not accept partial outputs", markdown)
            self.assertIn(str(error), markdown)

    def test_existing_directory_is_never_clobbered(self):
        directory = self.root / "existing"
        directory.mkdir()
        sentinel = directory / "run.log"
        sentinel.write_bytes(b"original contents\n")
        with (
            patch.object(cli_main, "extract") as extract,
            self.assertRaises(FileExistsError),
        ):
            cli_main.main(self.argv(directory))
        extract.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"original contents\n")
        self.assertEqual(list(directory.iterdir()), [sentinel])

    def test_invalid_arguments_are_reported_inside_inputs_stage(self):
        directory = self.root / "invalid"
        self.assertEqual(
            cli_main.main(self.argv(directory, "--two-theta-step-size", "0")), 1
        )
        report = strict_json(directory / "report.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["type"], "ValueError")
        self.assertEqual(report["stages"]["inputs"]["status"], "failed")
        self.assertFalse(list(directory.glob(".grouped_*")))


class QMaxInputTests(TemporaryCase):
    dataset = "/entry/plan_metadata/experiment_definition/data/q_max"

    def test_explicit_override_does_not_read_nexus(self):
        self.assertEqual(
            cli.resolve_q_max(self.root / "nonexistent.nxs", 20), (20, "cli")
        )

    def test_missing_and_empty_metadata_use_pyfai_maximum(self):
        path = self.root / "scan.nxs"
        for value in (None, h5py.Empty("f"), np.array([]), "", b"  "):
            with self.subTest(value=value):
                with h5py.File(path, "w") as source:
                    if value is not None:
                        source[self.dataset] = value
                self.assertEqual(cli.resolve_q_max(path), (None, "pyfai"))

    def test_scalar_singleton_and_string_metadata_in_angstrom_inverse(self):
        path = self.root / "scan.nxs"
        for value in (25.0, [25.0], "25.0", b"25.0"):
            with self.subTest(value=value):
                with h5py.File(path, "w") as source:
                    source[self.dataset] = value
                self.assertEqual(
                    cli.resolve_q_max(path), (25.0, f"nexus:{self.dataset}")
                )

    def test_nonempty_invalid_metadata_is_not_silently_ignored(self):
        path = self.root / "scan.nxs"
        for value in (0, -1, np.nan, np.inf, "nonsense", [20, 30]):
            with self.subTest(value=value):
                with h5py.File(path, "w") as source:
                    source[self.dataset] = value
                with self.assertRaises(ValueError):
                    cli.resolve_q_max(path)
                self.assertEqual(cli.resolve_q_max(path, 20.0), (20.0, "cli"))


class GeometryInputTests(TemporaryCase):
    def write_embedded(self, value):
        path = self.root / "scan.nxs"
        with h5py.File(path, "w") as source:
            source["entry/plan_metadata/geometry_calibration"] = value
        return path

    def test_embedded_scalar_and_singleton_json(self):
        config = {"wavelength": 1e-10, "detector_config": {"orientation": 3}}
        payload = json.dumps(config)
        for value in (
            payload,
            np.bytes_(payload),
            np.array([payload], dtype=h5py.string_dtype()),
        ):
            path = self.write_embedded(value)
            before = path.read_bytes()
            actual, provenance = cli.load_geometry(path)
            self.assertEqual(actual, config)
            self.assertEqual(provenance["geometry_source"], "nexus")
            self.assertEqual(
                provenance["geometry_dataset"],
                "/entry/plan_metadata/geometry_calibration",
            )
            self.assertIn(str(path.resolve()), provenance["geometry"])
            self.assertEqual(path.read_bytes(), before)
            external = self.root / "geometry.json"
            external.write_text(payload)
            _, file_provenance = cli.load_geometry(path, external)
            self.assertEqual(
                provenance["geometry_sha256"], file_provenance["geometry_sha256"]
            )

    def test_explicit_file_wins_without_opening_nexus(self):
        path = self.write_embedded("malformed embedded JSON")
        external = self.root / "override.json"
        external.write_text('{"chosen": "external"}')
        with patch.object(
            cli.h5py,
            "File",
            side_effect=AssertionError("Must not read embedded geometry"),
        ):
            actual, provenance = cli.load_geometry(path, external)
        self.assertEqual(actual, {"chosen": "external"})
        self.assertEqual(provenance["geometry_source"], "file")
        self.assertEqual(provenance["geometry"], str(external.resolve()))

    def test_invalid_explicit_file_never_falls_back(self):
        path = self.write_embedded('{"chosen": "embedded"}')
        with self.assertRaises(FileNotFoundError):
            cli.load_geometry(path, self.root / "missing.json")
        external = self.root / "override.json"
        for payload in ("invalid JSON", "[]", "null"):
            external.write_text(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                cli.load_geometry(path, external)

    def test_missing_group_or_invalid_embedded_blob(self):
        path = self.root / "scan.nxs"
        for kind in (
            "missing",
            "group",
            "invalid",
            "multiple",
            "numeric",
            "not_object",
        ):
            with h5py.File(path, "w") as source:
                if kind == "group":
                    source.create_group("entry/plan_metadata/geometry_calibration")
                elif kind not in ("missing",):
                    source["entry/plan_metadata/geometry_calibration"] = {
                        "invalid": "bad JSON",
                        "multiple": np.array([b"{}", b"{}"]),
                        "numeric": 42,
                        "not_object": "[]",
                    }[kind]
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "geometry|Geometry"),
            ):
                cli.load_geometry(path)


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.unity = processed_payload()
        self.gains = np.exp(np.array([-0.03, -0.01, 0.01, 0.03]))
        self.final = workflow.apply_position_scales(self.unity, self.gains)
        self.scale_fit = {"n_components": 1, "n_positions": 4}

    def test_conservation_identity_and_original_not_mutated(self):
        original = self.unity["groups"]["merged_2theta"]["per_frame"].copy()
        result = cli.check_conservation(self.unity, self.final, self.scale_fit)
        self.assertEqual(
            result, {"status": "passed", "supported_bins": 8, "unsupported_bins": 0}
        )
        np.testing.assert_array_equal(
            self.unity["groups"]["merged_2theta"]["per_frame"], original
        )
        np.testing.assert_array_equal(self.unity["frame_gains"], np.ones(4))

    def test_conservation_rejects_broken_identities(self):
        for field in (0, 1, 2, 3):
            with self.subTest(field=field):
                broken = copy.deepcopy(self.final)
                broken["groups"]["merged_2theta"]["per_frame"][0, field, 0] += 1
                with self.assertRaises(AssertionError):
                    cli.check_conservation(self.unity, broken, self.scale_fit)
        for graph in (
            {"n_components": 2, "n_positions": 4},
            {"n_components": 1, "n_positions": 3},
        ):
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                cli.check_conservation(self.unity, self.final, graph)
        ungauged = workflow.apply_position_scales(self.unity, np.full(4, 2.0))
        with self.assertRaises(AssertionError):
            cli.check_conservation(self.unity, ungauged, self.scale_fit)

    def test_unsupported_bins_are_counted(self):
        unity = copy.deepcopy(self.unity)
        group = unity["groups"]["merged_2theta"]
        group["per_frame"][:, :, -1] = 0
        group["accumulators"][:, -1] = 0
        group["intensity"][-1] = group["sigma"][-1] = np.nan
        final = workflow.apply_position_scales(unity, self.gains)
        self.assertEqual(
            cli.check_conservation(unity, final, self.scale_fit)["unsupported_bins"], 1
        )

    def test_diagnostic_leave_one_out_and_influence_identity(self):
        group = self.final["groups"]["merged_2theta"]
        result = diagnostics.position_overlap_diagnostics(group, min_pixel_weight=100)
        self.assertTrue(result["valid"].all())
        for index, frame in enumerate(group["per_frame"]):
            others = group["accumulators"] - frame
            own = frame[0] / frame[1]
            other = others[0] / others[1]
            residual = 100 * (own / other - 1)
            np.testing.assert_allclose(result["residual"][index], residual, atol=1e-12)
            # Influence is exactly the fractional change versus omitting a position.
            np.testing.assert_allclose(
                result["influence"][index],
                100 * (group["intensity"] / other - 1),
                atol=1e-12,
            )
        np.testing.assert_allclose(result["fraction"].sum(axis=0), 1.0)
        np.testing.assert_allclose(
            result["influence"], result["fraction"] * result["residual"]
        )

    def test_diagnostics_unsupported_and_invalid_inputs(self):
        group = copy.deepcopy(self.final["groups"]["merged_2theta"])
        group["per_frame"][:, :, -1] = 0
        group["accumulators"][:, -1] = 0
        result = diagnostics.position_overlap_diagnostics(group)
        self.assertTrue(np.isnan(result["residual"][:, -1]).all())
        self.assertTrue(np.isnan(result["influence"][:, -1]).all())
        np.testing.assert_array_equal(result["fraction"][:, -1], 0)
        self.assertFalse(result["valid"][:, -1].any())
        for weight in (0, -1, np.inf, np.nan):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                diagnostics.position_overlap_diagnostics(group, weight)
        with self.assertRaises(ValueError):
            diagnostics.position_overlap_diagnostics(
                {"per_frame": np.zeros((2, 3, 5)), "accumulators": np.zeros((4, 5))}
            )


class ThetaIntegrationTests(TemporaryCase):
    def test_one_theta_only_integration_per_position_and_count_valid_sampling(self):
        shape = (4, 6)
        counts = np.full((3, *shape), 2.0)
        counts[:, 0, :3] = [np.nan, -1.0, np.inf]
        counts[:, 0, 3] = 0.0  # Zero counts remain a valid observation.
        counts[:, 0, 4] = 1e9  # Already-grouped sums are not saturation-rejected.
        theta = np.array(
            [
                [12.0, 12.0, 12.0, 10.0, 20.0, 60.0],
                [20.0, 20.0, 20.0, 20.0, np.nan, 9.0],
                [61.0, 15.0, 25.0, 35.0, 45.0, 55.0],
                [10.0, 21.0, 31.0, 41.0, 51.0, 60.0],
            ]
        )
        path = self.root / "grouped.h5"
        with h5py.File(path, "x") as target:
            target["data"] = counts
        base = np.zeros(shape, dtype=bool)
        base[1, 0] = True
        fit = response_fit(shape, 3)
        fit["bad_pixel_mask"][1, 1] = True
        native = np.zeros(shape, dtype=bool)
        native[1, 2] = True
        azimuth = np.zeros(shape, dtype=bool)
        azimuth[1, 3] = True
        scan = stub(
            angles=np.array([1.0, 2.0, 3.0]),
            i0=np.array([10.0, 20.0, 30.0]),
            shape=shape,
            native_mask=native,
            detector_path=path,
            detector_dataset="data",
        )
        ai = stub(center_array=lambda shape, unit: np.deg2rad(theta))
        model = stub(
            geometry=stub(get_ai=lambda angle: ai, wavelength=1e-10),
            azimuth_range=(67.5, 112.5),
        )
        axis = np.arange(15.0, 60.0, 10.0)

        def integration(*args, **kwargs):
            return stub(
                radial=axis,
                sum_signal=np.full(5, 10.0),
                sum_normalization=np.ones(5),
                sum_variance=np.full(5, 10.0),
                count=np.full(5, 0.25),
            )

        with (
            patch.object(
                workflow, "integrate_frame", side_effect=integration
            ) as integrate,
            patch.object(workflow, "azimuth_mask", return_value=azimuth),
        ):
            result = cli.integrate_theta(
                scan, model, fit, base, (10.0, 60.0), 5, [2, 1, 3]
            )
        self.assertEqual(integrate.call_count, 3)
        self.assertEqual(set(result["groups"]), {"merged_2theta"})
        for index, call in enumerate(integrate.call_args_list):
            self.assertEqual(call.kwargs["unit"], "2th_deg")
            self.assertEqual(call.kwargs["saturation"], np.inf)
            self.assertEqual(call.args[2], scan.angles[index])
            self.assertEqual(call.args[3], scan.i0[index])
            np.testing.assert_array_equal(
                call.args[4], base | native | fit["bad_pixel_mask"]
            )
            np.testing.assert_array_equal(call.args[1], counts[index])
        bad = base | native | fit["bad_pixel_mask"] | azimuth
        valid = (
            ~bad
            & np.isfinite(counts[0])
            & (counts[0] >= 0)
            & np.isfinite(theta)
            & (theta >= 10)
            & (theta <= 60)
        )
        expected = np.histogram(theta[valid], bins=np.linspace(10, 60, 6))[0]
        sampling = result["sampling"]
        np.testing.assert_array_equal(
            sampling["observations_by_position"], np.tile(expected, (3, 1))
        )
        self.assertEqual(sampling["observations_by_position"].sum(), 3 * valid.sum())
        self.assertEqual(sampling["observations_by_position"].dtype.kind, "i")
        np.testing.assert_array_equal(sampling["group_sizes"], [2, 1, 3])
        self.assertNotEqual(float(expected.sum()), 5 * 0.25)
        group = result["groups"]["merged_2theta"]
        np.testing.assert_allclose(
            group["per_frame"].sum(axis=0), group["accumulators"]
        )


class FinalIntegrationTests(TemporaryCase):
    def test_independent_grids_frozen_gains_and_input_immutability(self):
        unity = processed_payload()
        unity["groups"]["merged_q"] = copy.deepcopy(unity["groups"]["merged_2theta"])
        original_q = unity["groups"]["merged_q"]["axis"].copy()
        gains = np.exp(np.array([-0.03, -0.01, 0.01, 0.03]))
        model = stub(geometry=stub(wavelength=1e-10))
        fit = response_fit()
        with patch.object(workflow, "process_scan", return_value=unity) as integrate:
            output = cli.integrate_output(
                stub(),
                model,
                fit,
                unity["mask"],
                (0.0, 60.0),
                0.025,
                0.025,
                gains,
                saturation=1e7,
            )
        options = integrate.call_args.kwargs
        self.assertEqual(options["theta_npt"], 2400)
        self.assertEqual(options["q_npt"], 252)
        self.assertEqual(options["saturation"], 1e7)
        np.testing.assert_allclose(options["q_range"], (0.0, 20 * np.pi))
        np.testing.assert_array_equal(unity["groups"]["merged_q"]["axis"], original_q)
        np.testing.assert_allclose(
            output["groups"]["merged_q"]["axis"], original_q / 10
        )
        self.assertEqual(output["groups"]["merged_q"]["unit"], "q_A^-1")
        for name, group in output["groups"].items():
            original = unity["groups"][name]["per_frame"]
            np.testing.assert_array_equal(
                group["per_frame"][:, (0, 2, 3)], original[:, (0, 2, 3)]
            )
            np.testing.assert_allclose(
                group["per_frame"][:, 1], original[:, 1] * gains[:, None]
            )


class DiagnosticSmokeTests(TemporaryCase):
    def test_all_jpegs_headless_supported_overlap_and_closed_figures(self):
        import matplotlib.pyplot as plt

        before_figures = plt.get_fignums()
        unity = processed_payload()
        shape = unity["mask"].shape
        fit = response_fit(shape)
        scan = stub(
            angles=np.array([5.0, 15.0, 25.0, 35.0]),
            i0=np.array([2.0, 3.0, 4.0, 5.0]),
            ring_current=np.array([280.0, 279.0, 278.0, 277.0]),
            shape=shape,
        )
        raw = stub(angles=np.repeat(scan.angles, 2), i0=np.repeat(scan.i0 / 2, 2))
        groups = [np.array([2 * i, 2 * i + 1]) for i in range(4)]
        model = stub(geometry=stub(wavelength=1e-10), azimuth_range=(67.5, 112.5))
        gains = np.exp(np.array([-0.03, -0.01, 0.01, 0.03]))
        final = workflow.apply_position_scales(unity, gains)
        scale = {
            "n_positions": 4,
            "n_components": 1,
            "n_edges": 6,
            "edge_rms_before": 0.02,
            "edge_rms_after": 0.001,
        }
        with (
            patch.object(
                workflow, "azimuth_mask", return_value=np.zeros(shape, dtype=bool)
            ),
            patch.object(workflow, "integrate_frame") as integrate,
            patch.object(
                workflow, "apply_position_scales", wraps=workflow.apply_position_scales
            ) as apply,
        ):
            summary = diagnostics.save_diagnostics(
                self.root,
                raw,
                scan,
                groups,
                model,
                unity["mask"],
                fit,
                unity,
                final,
                scale,
                theta_range=(30.0, 80.0),
                diagnostic_range=(30.0, 80.0),
                min_pixel_weight=100,
                dpi=35,
            )
        expected = {
            "angle_grouping",
            "azimuth_geometry",
            "detector_response",
            "sampling_q",
            "position_gains",
            "final_xy",
            "overlap_final_abcd",
        }
        self.assertEqual({Path(name).stem for name in summary["figures"]}, expected)
        self.assertEqual(summary["skips"], [])
        self.assertEqual(summary["warnings"], [])
        self.assertEqual(plt.get_fignums(), before_figures)
        self.assertEqual(str(plt.get_backend()).lower(), "agg")
        for name in summary["figures"]:
            image = self.root / name
            self.assertGreater(image.stat().st_size, 100)
            self.assertEqual(image.read_bytes()[:2], b"\xff\xd8")
            self.assertEqual(plt.imread(image).ndim, 3)
        self.assertEqual(summary["sampling"]["observations"], 4 * 8 * 3)
        self.assertEqual(summary["sampling"]["raw_frame_equivalents"], 2 * 4 * 8 * 3)
        self.assertTrue(
            all(row["overlap_bins"] == 8 for row in summary["residuals"]["final"])
        )
        json.dumps(summary, allow_nan=False)
        integrate.assert_not_called()
        apply.assert_not_called()
        self.assertEqual(set(summary["residuals"]), {"final"})
        expected_residuals = diagnostics.position_overlap_diagnostics(
            final["groups"]["merged_2theta"]
        )
        for row, residuals in zip(
            summary["residuals"]["final"], expected_residuals["residual"], strict=True
        ):
            self.assertAlmostEqual(row["mean_percent"], float(np.mean(residuals)))

    def test_figure_closed_if_jpeg_save_fails(self):
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        figures = plt.get_fignums()
        summary = {"figures": []}
        with patch.object(
            Figure, "savefig", side_effect=OSError("synthetic encoder failure")
        ):
            with self.assertRaises(OSError):
                with diagnostics._figure(self.root, summary, "broken", 35) as fig:
                    fig.subplots().plot([0, 1])
        self.assertEqual(plt.get_fignums(), figures)
        self.assertEqual(summary["figures"], [])


class SyntheticExtractionTests(TemporaryCase):
    """Run actual CLI orchestration and calibration/XY/processed HDF5 writers."""

    def synthetic_workflow(self, stack):
        # Calibration's public loader enforces the real detector shape. No real
        # frames or geometry calculations are needed to exercise its serializer.
        shape = (512, 1028)
        angles = np.array([10.0, 20.0, 30.0, 40.0])
        raw = workflow.Scan(
            self.root / "scan.nxs",
            self.root / "raw.h5",
            "data",
            np.repeat(angles, 2),
            np.ones(8),
            np.zeros(shape, dtype=bool),
            76.0,
            0.00075,
            "CdTe",
            1,
            1e-10,
            shape,
        )
        with h5py.File(raw.path, "x") as source:
            source["entry/instrument/synchrotron/current"] = np.full(8, 300.0)
        (self.root / "geometry.json").write_text("{}")
        model = stub(
            config={},
            polarization=0.95,
            density=6.2,
            energy=76.0,
            thickness=0.00075,
            material="CdTe",
            mu=2.0,
            transmission=0.8,
            efficiency=0.2,
            azimuth_range=(67.5, 112.5),
            geometry=stub(wavelength=1e-10),
            vertical_check=lambda: {"passed": True},
        )
        groups = [np.array([2 * i, 2 * i + 1]) for i in range(4)]

        def aggregate(source, path, **kwargs):
            # Exercise extraction's own TemporaryDirectory cleanup on both paths.
            with h5py.File(path, "x") as target:
                target["synthetic"] = np.arange(4)
            scan = copy.copy(raw)
            scan.angles, scan.i0, scan.detector_path = angles, np.full(4, 2.0), path
            return scan, groups

        fit = response_fit(shape)
        unity = processed_payload(shape=shape)
        scale = {
            "frame_gains": np.exp(np.array([-0.03, -0.01, 0.01, 0.03])),
            "n_components": 1,
            "n_positions": 4,
            "n_edges": 6,
            "settings": {},
            "gauge": "mean log gain zero",
        }
        mocks = {
            "load_scan": {"return_value": raw},
            "aggregate_equal_angle_frames": {"side_effect": aggregate},
            "PhysicalModel": {"return_value": model},
            "detector_layout": {
                "return_value": (
                    np.zeros(shape, dtype=bool),
                    np.zeros(shape, dtype=int),
                )
            },
            "constant_pattern_check": {"return_value": None},
            "build_cache": {
                "return_value": {i: {"pixels": np.arange(4)} for i in range(4)}
            },
            "fit_response": {"return_value": fit},
            "theta_range_for_scan": {"return_value": (30.0, 80.0)},
            "refine_position_scales": {"return_value": scale},
        }
        for name, options in mocks.items():
            stack.enter_context(patch.object(workflow, name, **options))
        stack.enter_context(patch.object(cli, "integrate_theta", return_value=unity))
        output = copy.deepcopy(unity)
        output["groups"]["merged_q"] = copy.deepcopy(unity["groups"]["merged_2theta"])
        stack.enter_context(patch.object(workflow, "process_scan", return_value=output))
        stack.enter_context(
            patch.object(
                diagnostics,
                "save_diagnostics",
                return_value={"figures": [], "warnings": []},
            )
        )
        return unity, fit

    def test_output_crop_does_not_change_scale_integration(self):
        directory = self.root / "cropped"
        with ExitStack() as stack:
            unity, _ = self.synthetic_workflow(stack)
            limits = cast(MagicMock, workflow.theta_range_for_scan)
            limits.return_value = (0.0, 100.0)
            q_max = 4 * np.pi * np.sin(np.deg2rad(35) / 2)
            code = cli_main.main(
                self.argv(directory, "--validation", "basic", "--q-max", str(q_max))
            )
            self.assertEqual(code, 0, (directory / "run.log").read_text())
            integration = cast(MagicMock, cli.integrate_theta)
            self.assertEqual(integration.call_args.args[4], (0.0, 100.0))
            self.assertNotIn("theta_max", limits.call_args_list[0].kwargs)
            scales = cast(MagicMock, workflow.refine_position_scales)
            self.assertIs(scales.call_args.args[0], unity["groups"]["merged_2theta"])
            output = cast(MagicMock, workflow.process_scan)
            np.testing.assert_allclose(
                output.call_args.kwargs["theta_range"], (0.0, 35.0)
            )
            np.testing.assert_allclose(
                output.call_args.kwargs["q_range"], (0.0, q_max * 10)
            )

    def test_nexus_limit_only_controls_final_output(self):
        directory = self.root / "nexus_crop"
        with ExitStack() as stack:
            self.synthetic_workflow(stack)
            with h5py.File(self.root / "scan.nxs", "r+") as source:
                source[QMaxInputTests.dataset] = 4.0
            code = cli_main.main(self.argv(directory, "--validation", "basic"))
            self.assertEqual(code, 0, (directory / "run.log").read_text())
            output = cast(MagicMock, workflow.process_scan)
            np.testing.assert_allclose(output.call_args.kwargs["q_range"], (0.0, 40.0))
            scales = cast(MagicMock, cli.integrate_theta)
            self.assertEqual(scales.call_args.args[4], (30.0, 80.0))
        report = strict_json(directory / "report.json")
        self.assertEqual(report["output_q_max_angstrom_inverse"], 4.0)
        self.assertEqual(
            report["output_q_max_source"], f"nexus:{QMaxInputTests.dataset}"
        )

    def test_q_limit_conversion_and_physical_bounds(self):
        full = (0.0, 85.0)
        self.assertEqual(cli.output_theta_range(None, 1e-10, full), full)
        np.testing.assert_allclose(
            cli.output_theta_range(2 * np.pi, 1e-10, full), (0, 60)
        )
        self.assertEqual(cli.output_theta_range(4 * np.pi, 1e-10, full), (0, 180))
        for value in (0, -1, np.inf, np.nan, 4 * np.pi + 0.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cli.output_theta_range(value, 1e-10, full)

    def test_end_to_end_synthetic_extraction_real_exports(self):
        directory = self.root / "extracted"
        with ExitStack() as stack:
            unity, fit = self.synthetic_workflow(stack)
            code = cli_main.main(
                self.argv(
                    directory,
                    "--validation",
                    "basic",
                    "--q-step-size",
                    "1.875",
                    "--two-theta-step-size",
                    "6.25",
                )
            )
            cache_builder = cast(MagicMock, workflow.build_cache)
            self.assertEqual(cache_builder.call_args.args[3], (100.0, 400.0))
        self.assert_exports(directory, code, unity, fit)
        with h5py.File(directory / "pattern.h5", "r") as target:
            entry = h5_group(target, str(target.attrs["default"]))
            self.assertEqual(entry.attrs["NX_class"], "NXentry")
            group = h5_group(entry, str(entry.attrs["default"]))
            self.assertEqual(group.name, "/entry/merged_2theta")
            self.assertEqual(group.attrs["NX_class"], "NXdata")
            self.assertEqual(group.attrs["signal"], "intensity")
            self.assertEqual(
                np.asarray(group.attrs["axes"]).tolist(), ["two_theta_deg"]
            )
            self.assertEqual(group.attrs["two_theta_deg_indices"], 0)
            self.assertEqual(group["two_theta_deg"].attrs["units"], "degrees")
            self.assertEqual(group["intensity"].attrs["interpretation"], "spectrum")
            self.assertEqual(group["intensity"].attrs["signal"], 1)
            self.assertEqual(group["intensity_errors"].id, group["sigma"].id)
            q_group = h5_group(entry, "merged_q")
            self.assertEqual(q_group.attrs["unit"], "q_A^-1")
            self.assertEqual(q_group["q_angstrom"].attrs["units"], "angstrom^-1")
            np.testing.assert_allclose(
                h5_dataset(q_group, "q_angstrom")[()],
                unity["groups"]["merged_2theta"]["axis"] / 10,
            )
            q_xy = np.loadtxt(directory / "pattern_q.xy")
            np.testing.assert_allclose(
                q_xy[:, 0], h5_dataset(q_group, "q_angstrom")[()], rtol=6e-10
            )
            np.testing.assert_allclose(
                q_xy[:, 1], h5_dataset(q_group, "intensity")[()], rtol=6e-10
            )
        with h5py.File(directory / "detector_corrections.h5", "r") as target:
            entry = h5_group(target, str(target.attrs["default"]))
            self.assertEqual(entry.attrs["NX_class"], "NXentry")
            group = h5_group(entry, str(entry.attrs["default"]))
            self.assertEqual(group.id, target["corrections"].id)
            self.assertEqual(group.attrs["NX_class"], "NXdata")
            self.assertEqual(group.attrs["signal"], "combined_divisor")
            axes = np.asarray(group.attrs["axes"]).tolist()
            self.assertEqual(axes, ["detector_row", "detector_column"])
            signal = h5_dataset(group, str(group.attrs["signal"]))
            self.assertEqual(signal.attrs["interpretation"], "image")
            self.assertEqual(signal.id, target["corrections/combined_divisor"].id)
            for dimension, axis in enumerate(axes):
                self.assertEqual(group.attrs[f"{axis}_indices"], dimension)
                axis_values = h5_dataset(group, axis)
                self.assertEqual(axis_values.shape, (signal.shape[dimension],))
                np.testing.assert_array_equal(
                    axis_values[()], np.arange(signal.shape[dimension])
                )

    def test_end_to_end_uses_embedded_geometry_by_default(self):
        directory = self.root / "embedded_extraction"
        config = {"chosen": "embedded", "wavelength": 1e-10}
        with ExitStack() as stack:
            unity, fit = self.synthetic_workflow(stack)
            with h5py.File(self.root / "scan.nxs", "r+") as source:
                source["entry/plan_metadata/geometry_calibration"] = json.dumps(config)
            argv = self.argv(
                directory,
                "--validation",
                "basic",
                "--q-step-size",
                "1.875",
                "--two-theta-step-size",
                "6.25",
            )
            option = argv.index("--geometry")
            del argv[option : option + 2]
            code = cli_main.main(argv)
            factory = cast(MagicMock, workflow.PhysicalModel)
            self.assertEqual(factory.call_args.args[0], config)
        self.assert_exports(directory, code, unity, fit)
        report = strict_json(directory / "report.json")
        self.assertEqual(report["inputs"]["geometry_source"], "nexus")
        self.assertIn(
            "/entry/plan_metadata/geometry_calibration",
            (directory / "report.md").read_text(),
        )

    def test_end_to_end_success_with_processed_writer_boundary_mock(self):
        # Keep a success-path regression independent of the processed
        # writer implementation. Calibration/XY remain real;
        # this boundary mock writes real, reopenable 2theta HDF5 datasets.
        def save_theta(path, scan, calibration, model, processed):
            group = processed["groups"]["merged_2theta"]
            with h5py.File(path, "x") as target:
                entry = target.create_group("entry/merged_2theta")
                for name, value in (
                    ("intensity", group["intensity"]),
                    ("sigma", group["sigma"]),
                    ("two_theta_deg", group["axis"]),
                    ("frame_accumulators", group["per_frame"]),
                ):
                    entry.create_dataset(name, data=value)

        directory = self.root / "successful_extraction"
        with ExitStack() as stack:
            unity, fit = self.synthetic_workflow(stack)
            writer = stack.enter_context(
                patch.object(workflow, "save_processed", side_effect=save_theta)
            )
            code = cli_main.main(
                self.argv(
                    directory,
                    "--validation",
                    "basic",
                    "--q-step-size",
                    "1.875",
                    "--two-theta-step-size",
                    "6.25",
                )
            )
        writer.assert_called_once()
        self.assert_exports(directory, code, unity, fit)

    def assert_exports(self, directory, code, unity, fit):
        report = strict_json(directory / "report.json")
        self.assertFalse(list(directory.glob(".grouped_*")))
        self.assertEqual(
            code, 0, report.get("error", {}).get("traceback", "Extraction failed")
        )
        self.assertEqual(report["status"], "completed_with_warnings")
        self.assertTrue(
            all(item["status"] == "passed" for item in report["stages"].values())
        )
        self.assertTrue(
            any("validation disabled" in item for item in report["warnings"])
        )
        calibration = workflow.load_calibration(directory / "detector_corrections.h5")
        np.testing.assert_array_equal(
            calibration["combined_divisor"], fit["combined_divisor"]
        )
        expected = workflow.apply_position_scales(unity, calibration["frame_gains"])[
            "groups"
        ]["merged_2theta"]
        with h5py.File(directory / "pattern.h5", "r") as target:
            group = h5_group(target, "entry/merged_2theta")
            np.testing.assert_allclose(
                h5_dataset(group, "intensity")[()], expected["intensity"]
            )
            np.testing.assert_allclose(
                h5_dataset(group, "frame_accumulators")[()].sum(axis=0),
                expected["accumulators"],
            )
        xy = np.loadtxt(directory / "pattern_tth.xy")
        np.testing.assert_allclose(xy[:, 0], expected["axis"], rtol=6e-10)
        np.testing.assert_allclose(xy[:, 1], expected["intensity"], rtol=6e-10)
        with h5py.File(directory / "detector_corrections.h5", "r") as target:
            np.testing.assert_array_equal(
                h5_dataset(target, "fit_diagnostics/detector_fit_nuisance_frame_gains")[
                    ()
                ],
                fit["frame_gains"],
            )

    def test_real_extraction_failure_cleans_grouped_file_and_reports_stage(self):
        directory = self.root / "fit_failure"
        with ExitStack() as stack:
            self.synthetic_workflow(stack)
            stack.enter_context(
                patch.object(
                    workflow,
                    "fit_response",
                    side_effect=RuntimeError("synthetic fit failure"),
                )
            )
            self.assertEqual(
                cli_main.main(self.argv(directory, "--validation", "basic")), 1
            )
        report = strict_json(directory / "report.json")
        self.assertEqual(report["stages"]["detector_fit"]["status"], "failed")
        self.assertEqual(report["stages"]["aggregation"]["status"], "passed")
        self.assertEqual(report["error"]["message"], "synthetic fit failure")
        self.assertFalse(list(directory.glob(".grouped_*")))
        self.assertFalse((directory / "pattern_tth.xy").exists())


if __name__ == "__main__":
    unittest.main()
