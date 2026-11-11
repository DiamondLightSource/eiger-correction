"""Azimuth-mask regressions without reading beamline data."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

import eiger_correction.eiger_workflow as workflow


def stub(**attributes: Any) -> Any:
    """Stand-in for Scan or PhysicalModel."""
    return SimpleNamespace(**attributes)


class AzimuthTests(unittest.TestCase):
    def model(self, chi_by_angle, limits=(67.5, 112.5)):
        shape = np.asarray(next(iter(chi_by_angle.values()))).shape

        def get_ai(angle):
            chi = np.asarray(chi_by_angle[float(angle)], dtype=float)

            def center_array(requested_shape, unit):
                self.assertEqual(requested_shape, shape)
                if unit == "chi_rad":
                    return np.deg2rad(chi)
                self.assertEqual(unit, "2th_rad")
                return np.deg2rad(np.arange(chi.size).reshape(shape) + 1.0)

            return stub(center_array=center_array)

        model = stub(
            shape=shape,
            azimuth_range=limits,
            geometry=stub(get_ai=get_ai),
            polarization=0.95,
        )
        model.arrays = lambda angle: (
            get_ai(angle),
            np.full(shape, 250.0),
            np.ones(shape),
            np.ones(shape),
            1.0,
        )
        return model

    def test_single_vertical_wedge_and_inclusive_edges(self):
        model = self.model({0.0: [[-90.0, 0.0, 67.5, 90.0, 112.5, 180.0, np.nan]]})
        np.testing.assert_array_equal(
            workflow.azimuth_mask(model, 0),
            [[True, True, False, False, False, True, True]],
        )

    def test_seam_crossing_and_full_circle(self):
        model = self.model(
            {0.0: [[-180.0, -170.0, -169.0, 0.0, 169.0, 170.0, 180.0, np.nan]]},
            (170.0, -170.0),
        )
        np.testing.assert_array_equal(
            workflow.azimuth_mask(model, 0),
            [[False, False, True, True, True, False, False, True]],
        )
        for limits in ((-180.0, 180.0), (180.0, -180.0)):
            model.azimuth_range = limits
            np.testing.assert_array_equal(
                workflow.azimuth_mask(model, 0), [[False] * 7 + [True]]
            )

    def test_none_and_legacy_model_need_no_geometry(self):
        for model in (
            stub(shape=(2, 3)),
            stub(shape=(2, 3), azimuth_range=None),
        ):
            np.testing.assert_array_equal(
                workflow.azimuth_mask(model, 123), np.zeros((2, 3), dtype=bool)
            )

    def test_invalid_ranges(self):
        for limits in (
            (90, 90),
            (0, 361),
            (np.nan, 90),
            (0, np.inf),
            (1,),
            ((1, 2),),
            "invalid",
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                workflow.validate_azimuth_range(limits)
        self.assertEqual(workflow.validate_azimuth_range([67.5, 112.5]), (67.5, 112.5))

    def test_masks_change_with_motor_angle(self):
        model = self.model({0.0: [[0.0, 90.0]], 40.0: [[90.0, 0.0]]})
        np.testing.assert_array_equal(workflow.azimuth_mask(model, 0), [[True, False]])
        np.testing.assert_array_equal(workflow.azimuth_mask(model, 40), [[False, True]])

    def test_cache_uses_position_dependent_mask_without_mutating_base(self):
        model = self.model(
            {
                angle: [[90.0, 0.0]] if angle % 2 == 0 else [[0.0, 90.0]]
                for angle in range(4)
            }
        )
        scan = stub(
            angles=np.arange(4.0),
            i0=np.ones(4),
            detector_path="unused.h5",
            detector_dataset="data",
        )
        base_mask = np.zeros(model.shape, dtype=bool)
        with patch.object(workflow.h5py, "File") as source:
            source.return_value.__enter__.return_value = {
                "data": np.ones((4, *model.shape))
            }
            cache = workflow.build_cache(scan, model, base_mask, npt=16)
        for index in range(4):
            np.testing.assert_array_equal(cache[index]["pixels"], [index % 2])
        self.assertFalse(base_mask.any())

    def test_integration_combines_fixed_azimuth_and_count_masks(self):
        model = self.model({0.0: [[90.0, 0.0, 90.0, 90.0, 90.0]]})
        integrator = model.geometry.get_ai(0)
        captured = {}

        def integrate(counts, npt, **kwargs):
            captured.update(kwargs)
            captured["counts"] = counts
            return "integrated"

        integrator.integrate1d = integrate
        model.arrays = lambda angle: (integrator, None, None, np.ones(model.shape), 1.0)
        fixed = np.array([[False, False, True, False, False]])
        result = workflow.integrate_frame(
            model,
            np.array([[10.0, 20.0, 30.0, np.nan, -1.0]]),
            0.0,
            1.0,
            fixed,
            np.ones(model.shape),
            16,
            (200, 400),
        )
        self.assertEqual(result, "integrated")
        np.testing.assert_array_equal(
            captured["mask"], [[False, True, True, True, True]]
        )
        np.testing.assert_array_equal(captured["counts"], [[10.0, 0.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(fixed, [[False, False, True, False, False]])

    def test_empty_wedge_fails_rather_than_silently_using_all_pixels(self):
        model = self.model({0.0: [[0.0, 0.0]]})
        with self.assertRaisesRegex(ValueError, "no unmasked pixels"):
            workflow.integrate_frame(
                model,
                np.ones(model.shape),
                0.0,
                1.0,
                np.zeros(model.shape, bool),
                np.ones(model.shape),
                16,
                (200, 400),
            )

    def test_automatic_theta_max_obeys_per_position_wedge(self):
        model = self.model({0.0: [[90.0, 0.0, 0.0]], 40.0: [[0.0, 90.0, 0.0]]})
        scan = stub(shape=model.shape, angles=np.array([0.0, 40.0]))
        limits = workflow.theta_range_for_scan(scan, model, np.zeros(model.shape, bool))
        np.testing.assert_allclose(limits, (0.0, 2.0))
        model.azimuth_range = None
        np.testing.assert_allclose(
            workflow.theta_range_for_scan(scan, model, np.zeros(model.shape, bool)),
            (0.0, 3.0),
        )

    def test_processing_masks_both_axes_and_keeps_metadata(self):
        model = self.model({0.0: [[90.0, 0.0]], 40.0: [[0.0, 90.0]]})
        model.geometry.wavelength = 1e-11
        scan = stub(
            shape=model.shape,
            angles=np.array([0.0, 40.0]),
            i0=np.ones(2),
            native_mask=np.zeros(model.shape, bool),
            detector_path="unused.h5",
            detector_dataset="data",
        )
        calibration = {
            "fit_settings": {"q_range": (200, 400)},
            "base_mask": scan.native_mask,
            "bad_pixel_mask": np.zeros(model.shape, bool),
            "combined_divisor": np.ones(model.shape),
        }

        def integrate(counts, npt, **kwargs):
            kept = ~kwargs["mask"]
            return stub(
                radial=np.arange(npt, dtype=float),
                sum_signal=np.full(npt, counts[kept].sum()),
                sum_normalization=np.full(npt, kept.sum(), dtype=float),
                sum_variance=np.full(npt, kwargs["variance"][kept].sum()),
                count=np.full(npt, kept.sum(), dtype=float),
            )

        original_arrays = model.arrays

        def arrays(angle):
            integrator, *rest = original_arrays(angle)
            integrator.integrate1d = integrate
            return integrator, *rest

        model.arrays = arrays
        with patch.object(workflow.h5py, "File") as source:
            source.return_value.__enter__.return_value = {
                "data": np.array([[[10.0, 1000.0]], [[2000.0, 20.0]]])
            }
            processed = workflow.process_scan(
                scan, calibration, model, q_npt=16, theta_npt=16
            )
        self.assertEqual(processed["azimuth_range_deg"], (67.5, 112.5))
        for group in processed["groups"].values():
            self.assertIn("within_fit_range", group)
            np.testing.assert_allclose(group["intensity"], 15.0)
            np.testing.assert_array_equal(group["per_frame"][:, 3], np.ones((2, 16)))
        scaled = workflow.apply_position_scales(processed, np.array([0.9, 1.1]))
        self.assertEqual(scaled["azimuth_range_deg"], (67.5, 112.5))

    def test_calibration_reload_restores_wedge_and_legacy_default(self):
        scan = stub(
            energy_kev=80.0,
            thickness_m=0.001,
            wavelength_m=1e-11,
            material="CdTe",
            orientation=2,
        )
        physical = {
            "energy_kev": 80.0,
            "sensor_thickness_m": 0.001,
            "wavelength_m": 1e-11,
            "material": "CdTe",
            "orientation": 2,
            "polarization_factor": 0.95,
            "density_g_cm3": 6.2,
            "normal_transmission": 0.1,
        }
        for settings, expected in (
            ({"azimuth_range_deg": [67.5, 112.5]}, [67.5, 112.5]),
            ({}, None),
        ):
            with (
                self.subTest(settings=settings),
                patch.object(workflow, "PhysicalModel") as factory,
            ):
                factory.return_value.transmission = 0.1
                workflow.model_for_processing(
                    scan,
                    {"physical": physical, "geometry": {}, "fit_settings": settings},
                )
                self.assertEqual(factory.call_args.kwargs["azimuth_range"], expected)


if __name__ == "__main__":
    unittest.main()
