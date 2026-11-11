"""Lightweight validation regressions using synthetic data, not beamline files."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

import eiger_correction.eiger_workflow as workflow


def stub(**attributes: Any) -> Any:
    """Stand-in for Scan or PhysicalModel."""
    return SimpleNamespace(**attributes)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.npt = 32
        self.shape = (8, self.npt)
        self.scan = stub(
            detector_path="unused.h5",
            detector_dataset="data",
            shape=self.shape,
            angles=np.array([10.0, 20.0]),
            i0=np.array([1.0, 2.0]),
        )
        self.base_mask = np.zeros(self.shape, dtype=bool)
        self.fit = {
            "combined_divisor": np.full(self.shape, 1.01),
            "bad_pixel_mask": np.zeros(self.shape, dtype=bool),
            "profile_log": np.zeros(self.npt),
            "supported_q_bins": np.ones(self.npt, dtype=bool),
        }
        bins = np.tile(np.arange(self.npt), self.shape[0])
        frame = {
            "pixels": np.arange(bins.size),
            "bins": bins,
            "counts": np.full(bins.size, 1000.0),
            "log_norm": np.full(bins.size, np.log(1000.5) - np.log(1.01) - 0.02),
        }
        self.cache = {index: frame.copy() for index in range(2)}

    def validate(self, **kwargs):
        return workflow.validate_response(
            self.scan,
            stub(),  # model is unused: integrate_frame is patched
            self.cache,
            self.fit,
            [0, 1],
            self.base_mask,
            (200.0, 400.0),
            self.npt,
            **kwargs,
        )

    def integration(self, *args, frame_gain, **kwargs):
        return stub(
            intensity=np.full(self.npt, np.exp(0.02) / frame_gain),
            count=np.full(self.npt, 100),
        )

    def test_disabled_skips_inference_and_lower_q_support(self):
        # Only 13 lower-Q pixels, and no inference inputs at all.
        self.cache = {index: {"bins": np.arange(self.npt)} for index in range(2)}
        intensity = np.ones(self.npt)
        intensity[:15] = 9.0  # Must not contaminate the higher-Q score.
        result = stub(intensity=intensity, count=np.full(self.npt, 100))
        with (
            patch.object(workflow.h5py, "File") as source,
            patch.object(workflow, "integrate_frame", return_value=result) as integrate,
            patch.object(
                workflow.np, "exp", side_effect=AssertionError("gain inference")
            ) as exp,
        ):
            source.return_value.__enter__.return_value = {
                "data": np.ones((2, *self.shape))
            }
            validation = self.validate(validate_frame_gains=False, saturation=1234.0)
            exp.assert_not_called()
        self.assertEqual(
            list(validation["results"]), ["physical_only", "detector_only"]
        )
        self.assertEqual(
            [row["label"] for row in validation["summary"]],
            ["physical_only", "detector_only"],
        )
        self.assertEqual(integrate.call_count, 4)
        for call in integrate.call_args_list:
            self.assertEqual(call.kwargs, {"frame_gain": 1.0, "saturation": 1234.0})
        for index, row in enumerate(validation["frame_rows"]):
            self.assertEqual(row["index"], index)
            self.assertEqual(row["angle_deg"], self.scan.angles[index])
            self.assertEqual(row["inferred_gain"], 1.0)
            self.assertEqual(row["cutoff_q_nm"], 278.125)
        for summary in validation["summary"]:
            self.assertEqual(summary["higher_q_rms_percent"], 0.0)

    def test_default_and_explicit_enabled_preserve_gain_validation(self):
        for kwargs in ({}, {"validate_frame_gains": True}):
            with (
                self.subTest(kwargs=kwargs),
                patch.object(workflow.h5py, "File") as source,
                patch.object(
                    workflow, "integrate_frame", side_effect=self.integration
                ) as integrate,
            ):
                source.return_value.__enter__.return_value = {
                    "data": np.ones((2, *self.shape))
                }
                validation = self.validate(**kwargs)
                self.assertEqual(
                    list(validation["results"]),
                    ["physical_only", "detector_only", "lower_q_frame_gain"],
                )
                self.assertEqual(integrate.call_count, 6)
                np.testing.assert_allclose(
                    [call.kwargs["frame_gain"] for call in integrate.call_args_list],
                    [1.0, 1.0, np.exp(0.02)] * 2,
                )
                np.testing.assert_array_equal(
                    integrate.call_args_list[0].args[5], np.ones(self.shape)
                )
                self.assertIs(
                    integrate.call_args_list[1].args[5], self.fit["combined_divisor"]
                )
                for row in validation["frame_rows"]:
                    self.assertAlmostEqual(row["inferred_gain"], np.exp(0.02))
                np.testing.assert_allclose(
                    [row["higher_q_rms_percent"] for row in validation["summary"]],
                    [2.0, 2.0, 0.0],
                    atol=1e-12,
                )

    def test_default_still_requires_lower_q_support(self):
        for frame in self.cache.values():
            for key, value in frame.items():
                frame[key] = value[: self.npt]
        with (
            patch.object(workflow.h5py, "File"),
            patch.object(workflow, "integrate_frame") as integrate,
        ):
            with self.assertRaisesRegex(ValueError, "Too few lower-q pixels"):
                self.validate()
            integrate.assert_not_called()

    def test_disabled_still_requires_higher_q_support(self):
        self.fit["supported_q_bins"][:] = False
        with (
            patch.object(workflow.h5py, "File") as source,
            patch.object(workflow, "integrate_frame", side_effect=self.integration),
        ):
            source.return_value.__enter__.return_value = {
                "data": np.ones((2, *self.shape))
            }
            with self.assertRaisesRegex(ValueError, "No supported higher-q bins"):
                self.validate(validate_frame_gains=False)


class FitResponseTests(unittest.TestCase):
    def test_disabled_fit_returns_unity_gains_for_eight_sensors(self):
        sensor_ids = np.broadcast_to(np.arange(8)[:, None], (8, 16)).copy()
        bins = np.tile(np.arange(16), 8)
        pixels = np.arange(sensor_ids.size)
        profile = np.linspace(0.0, 0.1, 16)
        sensor_log = np.linspace(-0.02, 0.02, 8)
        cache = {
            index: {
                "pixels": pixels,
                "bins": bins,
                "counts": np.exp(
                    7.0 + profile[bins] + sensor_log[sensor_ids.ravel()] + frame_log
                )
                - 0.5,
                "log_norm": np.zeros(pixels.size),
            }
            for index, frame_log in enumerate(np.linspace(-0.01, 0.01, 8))
        }
        with patch("builtins.print"):
            fit = workflow.fit_response(
                cache,
                np.arange(8),
                sensor_ids,
                frame_count=10,
                npt=16,
                iterations=3,
                fit_frame_gains=False,
            )
        np.testing.assert_array_equal(fit["frame_gains"], np.ones(10))
        np.testing.assert_array_equal(fit["observations"], np.full(sensor_ids.shape, 8))
        self.assertFalse(fit["fit_frame_gains"])
        self.assertEqual(len(fit["history"]), 3)
        self.assertTrue(np.all(np.isfinite(fit["combined_divisor"])))
        self.assertTrue(np.all(fit["combined_divisor"] > 0))


if __name__ == "__main__":
    unittest.main()
