"""Synthetic two-stage scalar overlap regressions; no beamline files required."""

from copy import deepcopy
import json
import unittest

import numpy as np

import eiger_workflow as workflow


def make_group(gains=None, coverage=None, bins=160):
    if gains is None:
        gains = np.array([0.7, 1.3, 1.8, 0.9])
    gains = np.asarray(gains, dtype=float)
    if coverage is None:
        coverage = [(0, 65), (25, 95), (55, 130), (90, 160)]
    axis = np.linspace(0.01, 15, bins)
    profile = 20 + 7 * np.sin(axis) ** 2
    frames = np.zeros((len(gains), 4, bins))
    for i, (start, stop) in enumerate(coverage):
        norm = (i + 1) * np.linspace(2, 5, bins)[start:stop]
        frames[i, 0, start:stop] = profile[start:stop] * gains[i] * norm
        frames[i, 1, start:stop] = norm
        frames[i, 2, start:stop] = frames[i, 0, start:stop] * (i + 1)
        frames[i, 3, start:stop] = 100 + 10 * i
    totals = frames.sum(axis=0)
    valid = totals[1] > 0
    intensity = np.divide(totals[0], totals[1], out=np.full(bins, np.nan), where=valid)
    sigma = np.divide(np.sqrt(totals[2]), totals[1], out=np.full(bins, np.nan), where=valid)
    return {"axis": axis, "per_frame": frames, "accumulators": totals,
            "intensity": intensity, "sigma": sigma,
            "within_fit_range": axis > 5}, gains, profile


def make_processed():
    theta, gains, profile = make_group()
    q, _, _ = make_group(bins=180, coverage=[(0, 70), (25, 100), (55, 135), (90, 170)])
    q["axis"] *= 10
    return {"groups": {"merged_2theta": theta, "merged_q": q},
            "frame_gains": np.ones(len(gains)), "mask": np.array([[True, False]]),
            "saturation": 123456., "custom": {"source": "synthetic"},
            "results": {"stale": object()}}, gains, profile


class RefinePositionScalesTests(unittest.TestCase):
    def test_connected_nonuniform_coverage_sign_gauge_and_diagnostics(self):
        group, gains, _ = make_group()
        baseline = deepcopy(group)
        result = workflow.refine_position_scales(group)
        expected = gains / np.exp(np.mean(np.log(gains)))
        np.testing.assert_allclose(result["frame_gains"], expected, rtol=2e-14)
        self.assertAlmostEqual(np.mean(np.log(result["frame_gains"])), 0., places=15)
        self.assertEqual([(edge["i"], edge["j"]) for edge in result["edges"]],
                         [(0, 1), (1, 2), (2, 3)])
        self.assertEqual([edge["n_bins"] for edge in result["edges"]], [40, 40, 40])
        self.assertEqual(result["components"], [[0, 1, 2, 3]])
        self.assertEqual(result["n_components"], 1)
        self.assertEqual(result["n_pair_bins"], 120)
        self.assertEqual(result["position_valid_bins"], [65, 70, 75, 70])
        self.assertEqual(result["position_edge_counts"], [1, 2, 2, 1])
        self.assertGreater(result["edge_rms_before"], 0.1)
        self.assertLess(result["edge_rms_after"], 1e-14)
        self.assertLess(result["weighted_edge_rms_after"], 1e-14)
        self.assertEqual(result["settings"], {"minimum_pixel_weight": 100.,
                         "minimum_overlap_bins": 20, "fractional_floor": 0.003})
        json.dumps({key: value for key, value in result.items() if key != "frame_gains"}, allow_nan=False)
        for key in group:
            np.testing.assert_array_equal(group[key], baseline[key])

    def test_all_pairs_joint_weighted_solution_with_noisy_cycle(self):
        group, _, _ = make_group([0.5, 1., 3.], [(0, 80)] * 3, bins=80)
        frames = group["per_frame"]
        frames[0, 0, :30] *= 1.04
        frames[1, 0, 20:70] *= 0.97
        frames[2, 2, :50] *= 10
        result = workflow.refine_position_scales(group, fractional_floor=0.01)
        incidence = np.array([[1., -1., 0.], [1., 0., -1.], [0., 1., -1.]])
        deltas, weights = [], []
        for edge, (i, j) in zip(result["edges"], [(0, 1), (0, 2), (1, 2)]):
            signal, norm, variance = frames[:, 0], frames[:, 1], frames[:, 2]
            w = 1 / (variance[i] / signal[i] ** 2 + variance[j] / signal[j] ** 2 + 0.01 ** 2)
            delta = np.average(np.log(signal[i] / norm[i]) - np.log(signal[j] / norm[j]), weights=w)
            self.assertAlmostEqual(edge["delta"], delta)
            self.assertAlmostEqual(edge["weight"], w.sum())
            deltas.append(delta)
            weights.append(w.sum())
        weights = np.array(weights)
        laplacian = incidence.T @ (weights[:, None] * incidence)
        constrained = np.block([[laplacian, np.ones((3, 1))], [np.ones((1, 3)), np.zeros((1, 1))]])
        expected = np.linalg.solve(constrained, np.r_[incidence.T @ (weights * deltas), 0.])[:3]
        np.testing.assert_allclose(np.log(result["frame_gains"]), expected, atol=1e-14)
        self.assertGreater(result["weighted_edge_rms_after"], 0.)
        self.assertLess(result["weighted_edge_rms_after"], result["weighted_edge_rms_before"])

    def test_invalid_noisy_samples_are_excluded_both_sides(self):
        group, gains, _ = make_group([0.8, 1.2], [(0, 40)] * 2, bins=40)
        frames = group["per_frame"]
        for field, value, position, index in [
            (0, 0., 0, 0), (0, -2., 1, 1), (0, np.nan, 0, 2),
            (0, np.inf, 1, 3), (2, -1., 0, 4), (2, np.inf, 1, 5),
            (1, 0., 0, 6), (1, np.nan, 1, 7), (3, 99., 0, 8),
            (3, np.nan, 1, 9), (3, 99., 1, 10), (1, -1., 1, 11),
        ]:
            frames[position, field, index] = value
        frames[:, 2, 12:15] = 0.  # Positive floor makes zero variance valid.
        result = workflow.refine_position_scales(group)
        self.assertEqual(result["edges"][0]["n_bins"], 28)
        np.testing.assert_allclose(result["frame_gains"], gains / np.exp(np.log(gains).mean()))

    def test_exact_support_thresholds_and_no_fit_range_restriction(self):
        group, _, _ = make_group([1., 2.], [(0, 20)] * 2, bins=20)
        group["per_frame"][:, 3] = 100.
        group["within_fit_range"][:] = False
        self.assertEqual(workflow.refine_position_scales(group)["n_pair_bins"], 20)
        group["per_frame"][1, 3, 0] = 99.999
        with self.assertRaisesRegex(ValueError, "Disconnected"):
            workflow.refine_position_scales(group)
        self.assertEqual(workflow.refine_position_scales(group, minimum_overlap_bins=19)["n_pair_bins"], 19)

    def test_disconnected_and_empty_unsupported_positions_fail(self):
        for coverage in ([(0, 40), (0, 40), (60, 100), (60, 100)],
                         [(0, 80), (0, 80), (0, 80), (0, 0)]):
            with self.subTest(coverage=coverage):
                group, _, _ = make_group(coverage=coverage)
                with self.assertRaisesRegex(ValueError, "Disconnected overlap graph.*components=.*unsupported positions="):
                    workflow.refine_position_scales(group)

    def test_invalid_parameters(self):
        group, _, _ = make_group()
        for name, values in {
            "minimum_pixel_weight": [0, -1, np.nan, np.inf, True, "100", [100]],
            "minimum_overlap_bins": [0, -1, 1.5, 20., np.nan, True, "20"],
            "fractional_floor": [0, -1, np.nan, np.inf, True, "0.003", 1e-200, 1e200],
        }.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    workflow.refine_position_scales(group, **{name: value})

    def test_invalid_shapes_axes_and_single_position(self):
        group, _, _ = make_group()
        invalid = [{}, {**group, "per_frame": np.zeros((4, 3, 160))},
                   {**group, "per_frame": np.zeros((0, 4, 160))},
                   {**group, "per_frame": np.zeros((4, 4, 0))},
                   {**group, "per_frame": np.zeros((4, 160))},
                   {**group, "axis": np.zeros(159)},
                   {**group, "axis": np.full(160, np.nan)},
                   {**group, "per_frame": group["per_frame"][:1]}]
        for payload in invalid:
            with self.subTest(keys=list(payload)), self.assertRaises(ValueError):
                workflow.refine_position_scales(payload)


class ApplyPositionScalesTests(unittest.TestCase):
    def test_normalization_only_merging_conservation_metadata_and_no_mutation(self):
        processed, gains, _ = make_processed()
        baseline = deepcopy(processed)
        for group in processed["groups"].values():
            group["results"] = {"stale": object()}
            group["per_frame"].setflags(write=False)
        supplied = gains.copy()
        corrected = workflow.apply_position_scales(processed, supplied)
        self.assertIsNot(corrected, processed)
        self.assertNotIn("results", corrected)
        self.assertIn("results", processed)
        self.assertEqual(corrected["saturation"], 123456.)
        self.assertEqual(corrected["custom"], processed["custom"])
        np.testing.assert_array_equal(corrected["mask"], processed["mask"])
        np.testing.assert_array_equal(processed["frame_gains"], np.ones(4))
        for name, original in processed["groups"].items():
            group = corrected["groups"][name]
            self.assertNotIn("results", group)
            self.assertIn("results", original)
            np.testing.assert_array_equal(original["per_frame"], baseline["groups"][name]["per_frame"])
            np.testing.assert_array_equal(group["per_frame"][:, [0, 2, 3]], original["per_frame"][:, [0, 2, 3]])
            np.testing.assert_array_equal(group["per_frame"][:, 1], original["per_frame"][:, 1] * gains[:, None])
            np.testing.assert_array_equal(group["accumulators"], group["per_frame"].sum(axis=0))
            np.testing.assert_array_equal(group["accumulators"][[0, 2, 3]], original["accumulators"][[0, 2, 3]])
            totals = group["accumulators"]
            valid = totals[1] > 0
            np.testing.assert_allclose(group["intensity"][valid], totals[0, valid] / totals[1, valid])
            np.testing.assert_allclose(group["sigma"][valid], np.sqrt(totals[2, valid]) / totals[1, valid])
            self.assertTrue(np.all(np.isnan(group["intensity"][~valid])))
            self.assertTrue(np.all(np.isnan(group["sigma"][~valid])))
            for field in ("axis", "within_fit_range"):
                np.testing.assert_array_equal(group[field], original[field])
            for field in ("intensity", "sigma", "accumulators"):
                np.testing.assert_array_equal(original[field], baseline["groups"][name][field])
        supplied[:] = 99
        np.testing.assert_array_equal(corrected["frame_gains"], gains)
        with self.assertRaisesRegex(ValueError, "exactly unity"):
            workflow.apply_position_scales(corrected, gains)

    def test_end_to_end_common_scale_including_nonoverlap_bins(self):
        processed, gains, profile = make_processed()
        fit = workflow.refine_position_scales(processed["groups"]["merged_2theta"])
        corrected = workflow.apply_position_scales(processed, fit["frame_gains"])
        group = corrected["groups"]["merged_2theta"]
        expected = profile * np.exp(np.mean(np.log(gains)))
        np.testing.assert_allclose(group["intensity"], expected, rtol=2e-14)
        for frame in group["per_frame"]:
            valid = frame[1] > 0
            np.testing.assert_allclose(frame[0, valid] / frame[1, valid], expected[valid], rtol=2e-14)

    def test_invalid_gains_and_baseline_provenance(self):
        processed, _, _ = make_processed()
        for gains in ([1., 2.], [[1.] * 4], [1., 1., 1., 0.], [1., 1., 1., -1.],
                      [1., 1., 1., np.nan], [1., 1., 1., np.inf], [], "bad"):
            with self.subTest(gains=gains), self.assertRaises(ValueError):
                workflow.apply_position_scales(processed, gains)
        for original in ([1., 1., 1., 1. + 1e-14], [1., 1.], [[1.] * 4], [1., 1., 1., np.nan]):
            processed["frame_gains"] = original
            with self.subTest(original=original), self.assertRaisesRegex(ValueError, "exactly unity"):
                workflow.apply_position_scales(processed, np.ones(4))

    def test_group_validation_and_failure_does_not_partially_mutate(self):
        for field, value in ((0, np.nan), (1, -1.), (2, -1.), (3, -1.)):
            processed, gains, _ = make_processed()
            processed["groups"]["merged_q"]["per_frame"][0, field, 0] = value
            before = deepcopy(processed["groups"])
            with self.subTest(field=field), self.assertRaises(ValueError):
                workflow.apply_position_scales(processed, gains)
            for name in before:
                np.testing.assert_array_equal(processed["groups"][name]["per_frame"], before[name]["per_frame"])
        processed, gains, _ = make_processed()
        processed["groups"]["merged_q"]["per_frame"] = np.zeros((3, 4, 180))
        with self.assertRaisesRegex(ValueError, "position count"):
            workflow.apply_position_scales(processed, gains)
        processed["groups"] = {}
        with self.assertRaisesRegex(ValueError, "nonempty dict"):
            workflow.apply_position_scales(processed, gains)


if __name__ == "__main__":
    unittest.main()