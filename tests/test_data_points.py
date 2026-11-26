"""Multi-point (blower temperature) scans: partitioning, subsetting and exports.

No beamline data are used. Scientific fitting and geometry are mocked through
the synthetic single-point workflow; partitioning, frame mapping, combined
HDF5, XY and report generation remain real.
"""

import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import h5py
import numpy as np

import eiger_correction.__main__ as cli_main
import eiger_correction.eiger_workflow as workflow
import eiger_correction.extract_eiger_corrections as cli

# A module import (not a class import) so pytest does not collect those tests here.
import test_extract_eiger_corrections as synthetic
from eiger_correction.eiger_report import render_markdown
from eiger_correction.eiger_workflow import h5_dataset, h5_group

strict_json = synthetic.strict_json

ANGLES = np.array([0.0, 0.0, 8.2878, 8.2878, 11.7186, 11.7186])


def specification(*setpoints, frames=2, angles=(0.0, 8.2878, 11.7186)):
    return json.dumps(
        [
            {
                "temperature_celsius": setpoint,
                "temperature_unit": "C",
                "collection_specification": [
                    {"tth": angle, "frames": frames} for angle in angles
                ],
            }
            for setpoint in setpoints
        ]
    )


class FindDataPointsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="eiger_points_")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "scan.nxs"

    def write(self, readings=None, spec=None):
        with h5py.File(self.path, "w") as source:
            source.create_group("entry/fastcs_eiger")
            if readings is not None:
                source[cli.POINT_AXIS_PATH] = readings
            if spec is not None:
                source[cli.COLLECTION_SPECIFICATION_PATH] = spec

    def test_single_point_scan_has_no_data_points(self):
        self.write()
        self.assertIsNone(cli.find_data_points(self.path, ANGLES))

    def test_collection_specification_partitions_frames(self):
        angles = np.tile(ANGLES, 2)
        angles[7] += 0.0001  # motor readback jitter within tolerance
        readings = np.array([49.5, 50.5, 50, 50, 49, 51, 99, 101, 100, 100, 98, 102])
        self.write(readings, specification(50.0, 100.0))
        points = cli.find_data_points(self.path, angles)
        assert points is not None
        self.assertEqual(
            [p["label"] for p in points], ["point_01_50C", "point_02_100C"]
        )
        self.assertEqual([(p["start"], p["stop"]) for p in points], [(0, 6), (6, 12)])
        self.assertEqual([p["setpoint"] for p in points], [50.0, 100.0])
        self.assertEqual(points[0]["partition"], "collection_specification")
        self.assertAlmostEqual(points[0]["measured_mean"], 50.0)
        self.assertEqual(points[1]["measured_min"], 98.0)
        self.assertEqual(points[1]["measured_max"], 102.0)
        np.testing.assert_array_equal(points[1]["readings"], readings[6:])

    def test_specification_must_match_recorded_frames(self):
        angles = np.tile(ANGLES, 2)
        self.write(np.full(12, 50.0), specification(50.0, 100.0, 150.0))
        with self.assertRaisesRegex(ValueError, "plans 18 frames"):
            cli.find_data_points(self.path, angles)
        angles[4] = 20.0
        self.write(np.full(12, 50.0), specification(50.0, 100.0))
        with self.assertRaisesRegex(ValueError, "frame 4"):
            cli.find_data_points(self.path, angles)
        self.write(np.full(10, 50.0), specification(50.0, 100.0))
        with self.assertRaisesRegex(ValueError, "10 readings for 12 frames"):
            cli.find_data_points(self.path, angles)

    def test_fallback_splits_where_two_theta_restarts(self):
        angles = np.tile(ANGLES, 3)
        self.write(np.repeat([25.0, 50.0, 75.0], 6), json.dumps([{"tth": 0}]))
        points = cli.find_data_points(self.path, angles)
        assert points is not None
        self.assertEqual(
            [(p["start"], p["stop"]) for p in points], [(0, 6), (6, 12), (12, 18)]
        )
        self.assertEqual(points[2]["label"], "point_03")
        self.assertEqual(points[2]["partition"], "2theta_sweep_restart")
        self.assertEqual(points[2]["measured_mean"], 75.0)

    def test_labels_are_filesystem_safe(self):
        self.assertEqual(cli.point_label(0, None), "point_01")
        self.assertEqual(cli.point_label(9, 50.0), "point_10_50C")
        self.assertEqual(cli.point_label(1, -12.5), "point_02_m12p5C")


class SubsetScanTests(unittest.TestCase):
    def test_subset_aggregation_reads_its_own_source_frames(self):
        with tempfile.TemporaryDirectory(prefix="eiger_subset_") as temporary:
            root = Path(temporary)
            shape = (2, 3)
            frames = np.arange(8, dtype=float)[:, None, None] * np.ones(shape)
            with h5py.File(root / "raw.h5", "w") as source:
                source["data"] = frames
            raw = workflow.Scan(
                root / "scan.nxs",
                root / "raw.h5",
                "data",
                np.tile([10.0, 10.0, 20.0, 20.0], 2),
                np.arange(1.0, 9.0),
                np.zeros(shape, dtype=bool),
                76.0,
                0.00075,
                "CdTe",
                1,
                1e-10,
                shape,
                ring_current=np.arange(8.0),
            )
            subset = workflow.subset_scan(raw, np.arange(4, 8))
            np.testing.assert_array_equal(subset.source_frames, [4, 5, 6, 7])
            np.testing.assert_array_equal(subset.ring_current, [4, 5, 6, 7])
            nested = workflow.subset_scan(subset, [1, 3])
            np.testing.assert_array_equal(nested.source_frames, [5, 7])
            grouped, groups = workflow.aggregate_equal_angle_frames(
                subset, root / "grouped.h5"
            )
            self.assertIsNone(grouped.source_frames)
            np.testing.assert_array_equal(grouped.i0, [5 + 6, 7 + 8])
            with h5py.File(root / "grouped.h5", "r") as target:
                np.testing.assert_array_equal(
                    h5_dataset(target, "data")[:, 0, 0], [4 + 5, 6 + 7]
                )
                np.testing.assert_array_equal(
                    h5_dataset(target, "source_frame_indices")[()], [4, 5, 6, 7]
                )
            self.assertEqual([g.tolist() for g in groups], [[0, 1], [2, 3]])
            for invalid in ([], [8], [-1], [1, 1]):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    workflow.subset_scan(raw, invalid)


class MultiPointExtractionTests(synthetic.TemporaryCase):
    """End-to-end CLI run of a two-temperature scan with mocked science."""

    def multi_point(self, stack):
        unity, fit = synthetic.SyntheticExtractionTests.synthetic_workflow(
            cast(synthetic.SyntheticExtractionTests, self), stack
        )
        loader = cast(MagicMock, workflow.load_scan)
        raw = loader.return_value
        # Two temperature points, each the synthetic 8-frame sweep.
        loader.return_value = replace(
            raw, angles=np.tile(raw.angles, 2), i0=np.ones(16), ring_current=None
        )
        readings = np.concatenate((np.full(8, 50.2), np.full(8, 99.8)))
        with h5py.File(self.root / "scan.nxs", "r+") as source:
            source[cli.POINT_AXIS_PATH] = readings
            source[cli.COLLECTION_SPECIFICATION_PATH] = specification(
                50.0, 100.0, angles=(10.0, 20.0, 30.0, 40.0)
            )
        return unity, fit

    def test_two_points_give_two_patterns_in_one_hdf5(self):
        directory = self.root / "blower"
        with ExitStack() as stack:
            unity, fit = self.multi_point(stack)
            code = cli_main.main(self.argv(directory, "--validation", "basic"))
            aggregate = cast(MagicMock, workflow.aggregate_equal_angle_frames)
            sources = [call.args[0].source_frames for call in aggregate.call_args_list]
        report = strict_json(directory / "report.json")
        self.assertEqual(code, 0, report.get("error", {}).get("traceback"))
        np.testing.assert_array_equal(sources[0], np.arange(8))
        np.testing.assert_array_equal(sources[1], np.arange(8, 16))
        labels = ["point_01_50C", "point_02_100C"]
        self.assertEqual([p["label"] for p in report["data_points"]], labels)
        self.assertEqual(
            [p["status"] for p in report["data_points"]], ["completed"] * 2
        )
        self.assertTrue(all(s["status"] == "passed" for s in report["stages"].values()))
        self.assertIn("point_02_100C/combine_pattern", report["stages"])
        self.assertTrue(
            any(w.startswith("point_02_100C: Independent") for w in report["warnings"])
        )
        self.assertFalse((directory / "pattern.h5").exists())
        self.assertFalse(list(directory.glob(".grouped_*")))
        self.assertFalse(list(directory.glob(".pattern_*")))
        with h5py.File(directory / "patterns.h5", "r") as target:
            self.assertEqual(sorted(target), labels)
            self.assertEqual(target.attrs["default"], labels[0])
            self.assertEqual(target.attrs["data_point_axis"], "blower-temperature")
            for label, setpoint, temperature in zip(
                labels, (50.0, 100.0), (50.2, 99.8), strict=True
            ):
                entry = h5_group(target, label)
                self.assertEqual(entry.attrs["NX_class"], "NXentry")
                self.assertEqual(entry.attrs["default"], "merged_2theta")
                group = h5_group(entry, "merged_2theta")
                self.assertEqual(group.attrs["NX_class"], "NXdata")
                calibration = workflow.load_calibration(
                    directory / label / "detector_corrections.h5"
                )
                expected = workflow.apply_position_scales(
                    unity, calibration["frame_gains"]
                )["groups"]["merged_2theta"]
                np.testing.assert_allclose(
                    h5_dataset(group, "intensity")[()], expected["intensity"]
                )
                point = h5_group(entry, "data_point")
                self.assertEqual(h5_dataset(point, "setpoint")[()], setpoint)
                np.testing.assert_allclose(
                    np.asarray(point.attrs["measured_mean"]), temperature
                )
                xy = np.loadtxt(directory / f"{label}_tth.xy")
                np.testing.assert_allclose(xy[:, 1], expected["intensity"], rtol=6e-10)
                self.assertTrue((directory / f"{label}_q.xy").exists())
                header = (directory / f"{label}_tth.xy").read_text().splitlines()[0]
                self.assertIn(f"setpoint {setpoint:g} C", header)
        np.testing.assert_array_equal(
            workflow.load_calibration(
                directory / labels[0] / "detector_corrections.h5"
            )["combined_divisor"],
            fit["combined_divisor"],
        )
        text = (directory / "report.md").read_text()
        self.assertIn("## Data points", text)
        self.assertIn("| point_02_100C | 100 °C | 99.8 ± 0 °C |", text)
        self.assertIn("[point_01_50C_tth.xy](point_01_50C_tth.xy)", text)

    def test_failure_in_second_point_is_reported(self):
        directory = self.root / "blower_failure"
        with ExitStack() as stack:
            self.multi_point(stack)
            fitter = cast(MagicMock, workflow.fit_response)
            fitter.side_effect = [fitter.return_value, RuntimeError("second point")]
            code = cli_main.main(self.argv(directory, "--validation", "basic"))
        self.assertEqual(code, 1)
        report = strict_json(directory / "report.json")
        self.assertEqual(
            [p["status"] for p in report["data_points"]], ["completed", "failed"]
        )
        self.assertEqual(
            report["stages"]["point_02_100C/detector_fit"]["status"], "failed"
        )
        self.assertFalse(list(directory.glob(".grouped_*")))
        self.assertFalse(list(directory.glob(".pattern_*")))
        self.assertIn("FAILED", (directory / "report.md").read_text())


class DataPointReportTests(unittest.TestCase):
    def test_per_point_tables_and_galleries(self):
        point = {
            "label": "point_01_50C",
            "setpoint": 50.0,
            "unit": "C",
            "start": 0,
            "stop": 138,
            "raw_frames": 138,
            "measured_mean": 50.04,
            "measured_std": 0.92,
            "measured_min": 48.4,
            "measured_max": 51.8,
            "status": "completed",
            "aggregation": {"positions": 46, "summed_i0": [1.5, 2.5]},
            "theta_range_deg": [0.0, 60.0],
            "integrity_checks": {
                "status": "passed",
                "supported_bins": 2401,
                "unsupported_bins": 0,
            },
            "position_scale_refinement": {
                "n_positions": 46,
                "n_edges": 702,
                "n_components": 1,
            },
            "held_out_validation": [
                {
                    "fold": 1,
                    "metrics": [
                        {"label": "physical_only", "absolute_overlap_percent": 0.66},
                        {"label": "detector_only", "absolute_overlap_percent": 0.33},
                    ],
                }
            ],
            "raw_audit": {"status": "passed"},
            "warnings": ["one"],
            "outputs": {"pattern": "patterns.h5::/point_01_50C"},
            "diagnostics": {"figures": ["point_01_50C/final_xy.jpg"]},
        }
        text = render_markdown(
            {
                "status": "completed",
                "inputs": {"data_point_axis": "blower-temperature", "raw_frames": 276},
                "data_points": [point, {"label": "point_02_100C", "status": "pending"}],
            }
        )
        for expected in (
            "| Data points | 2 |",
            "| point_01_50C | 50 °C | 50.04 ± 0.92 °C | 48.4 – 51.8 °C | "
            "138 (0–137) | 46 | 4 | completed |",
            "| point_01_50C | 0 – 60 | — | 2401 / 0 | 46 / 702 / 1 | — → — | "
            "0.66 → 0.33 | passed | passed | 1 |",
            "| point_02_100C | — | — | — | — (—–—) | — | — | pending |",
            "| point_01_50C | 1 | detector_only |",
            "### point_01_50C",
            "#### Final intensity pattern",
            "![Final intensity pattern — full range and high-angle zoom]"
            "(point_01_50C/final_xy.jpg)",
            "[patterns.h5::/point_01_50C](patterns.h5)",
            "point_01_50C: passed; point_02_100C: not yet recorded",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("Raw frames → analysis positions", text)


if __name__ == "__main__":
    unittest.main()
