"""Raw EIGER calibration and processing routines for the extraction CLI."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np
import pyFAI
import xraylib
from pyFAI.goniometer import Goniometer

SCHEMA = "eiger_joint_calibration_v1"
METHOD = ("bbox", "csr", "cython")
UNCERTAINTY = "Raw-count Poisson propagation only; excludes calibration, I0, model uncertainty and bin covariance"


def _array(file, path):
    dataset = file[path]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{path} is not a dataset")
    return np.asarray(dataset[()])


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass
class Scan:
    path: Path
    detector_path: Path
    detector_dataset: str
    angles: np.ndarray
    i0: np.ndarray
    native_mask: np.ndarray
    energy_kev: float
    thickness_m: float
    material: str
    orientation: int
    wavelength_m: float
    shape: tuple[int, int]


def load_scan(path, detector_path=None, monitor_path=None, metadata_path=None):
    path = Path(path).resolve()
    detector_path = (
        Path(detector_path)
        if detector_path
        else path.with_name(path.stem + "-fastcs_eiger_000001.h5")
    )
    monitor_path = (
        Path(monitor_path) if monitor_path else path.with_name(path.stem + "-i0.h5")
    )
    metadata_path = (
        Path(metadata_path)
        if metadata_path
        else path.with_name(path.stem + "-fastcs_eiger_meta.h5")
    )
    detector_dataset = "data"
    with h5py.File(path, "r") as file:
        angles = _array(file, "entry/fastcs_eiger/tth").astype(float).reshape(-1)
        calibration = file["entry/plan_metadata/geometry_calibration"]
        if isinstance(calibration, h5py.Dataset):
            calibration_config = json.loads(_text(calibration[()]))
            orientation = int(calibration_config["detector_config"]["orientation"])
            wavelength = float(calibration_config["wavelength"])
        else:
            orientation = int(_array(calibration, "detector_config/orientation"))
            wavelength = float(_array(calibration, "wavelength"))
        nexus_mask = (
            _array(file, "entry/instrument/fastcs_eiger/pixel_mask")
            if "entry/instrument/fastcs_eiger/pixel_mask" in file
            else None
        )
        if not detector_path.exists():
            detector_path, detector_dataset = path, "entry/fastcs_eiger/data"
    with h5py.File(monitor_path, "r") as file:
        i0 = _array(file, "entry/data/data").astype(float).reshape(-1)
    with h5py.File(metadata_path, "r") as file:
        mask = _array(file, "mask") != 0
        energy = float(_array(file, "_dectris/incident_energy").reshape(-1)[0]) / 1000
        thickness = float(_array(file, "_dectris/sensor_thickness").reshape(-1)[0])
        material = _text(_array(file, "_dectris/sensor_material").reshape(-1)[0])
    with h5py.File(detector_path, "r") as file:
        shape = file[detector_dataset].shape
    if (
        shape != (len(angles), 512, 1028)
        or len(i0) != len(angles)
        or mask.shape != shape[1:]
    ):
        raise ValueError(
            "Expected matching frame, angle, I0 and mask arrays for EIGER2 CdTe 500K (512,1028)"
        )
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(i0) & (i0 > 0)):
        raise ValueError(
            "Angles must be finite and every I0 must be finite and positive"
        )
    if material != "CdTe" or energy <= 0 or thickness <= 0:
        raise ValueError(
            "This workflow requires positive energy/thickness and CdTe metadata"
        )
    if nexus_mask is not None and nexus_mask.shape == mask.shape:
        mask |= nexus_mask != 0
    return Scan(
        path,
        detector_path.resolve(),
        detector_dataset,
        angles,
        i0,
        mask,
        energy,
        thickness,
        material,
        orientation,
        wavelength,
        shape[1:],
    )


def aggregate_equal_angle_frames(
    scan, output_path, saturation=1e7, angle_tolerance=0.0
):
    """Sum equal or tolerance-matched angle frames and I0 into a compact file.

    A detector pixel is retained in an aggregate only when every constituent
    frame has a finite, nonnegative, nonsaturated value at that pixel. This
    avoids treating a missing/saturated frame as a zero-count measurement.
    Group angles are arithmetic means of motor readings (one per frame),
    independent of I0; summed I0 is used only for intensity normalization.
    The original scan files are never modified.
    """
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite aggregated data: {output_path}")
    if not np.isfinite(saturation) or saturation <= 0:
        raise ValueError("saturation must be finite and positive")
    if not np.isfinite(angle_tolerance) or angle_tolerance < 0:
        raise ValueError("angle_tolerance must be finite and nonnegative")

    unique_angles, inverse = np.unique(scan.angles, return_inverse=True)
    angle_clusters = []
    for angle_index, angle in enumerate(unique_angles):
        if (
            not angle_clusters
            or angle - unique_angles[angle_clusters[-1][0]] > angle_tolerance
        ):
            angle_clusters.append([angle_index])
        else:
            angle_clusters[-1].append(angle_index)
    cluster_for_angle = np.empty(len(unique_angles), dtype=np.int32)
    for cluster_index, cluster in enumerate(angle_clusters):
        cluster_for_angle[cluster] = cluster_index
    group_for_frame = cluster_for_angle[inverse]
    group_members = [
        np.flatnonzero(group_for_frame == group_index)
        for group_index in range(len(angle_clusters))
    ]
    grouped_i0 = np.array(
        [scan.i0[members].sum() for members in group_members], dtype=float
    )
    grouped_angles = np.array(
        [np.mean(scan.angles[members]) for members in group_members], dtype=float
    )
    if not np.all(np.isfinite(grouped_i0) & (grouped_i0 > 0)):
        raise ValueError("Aggregated I0 values must be finite and positive")

    with (
        h5py.File(scan.detector_path, "r") as source,
        h5py.File(output_path, "x") as target,
    ):
        source_data = source[scan.detector_dataset]
        aggregated = target.create_dataset(
            "data",
            shape=(len(angle_clusters), *scan.shape),
            dtype=np.float64,
            chunks=(1, *scan.shape),
            compression="lzf",
        )
        for group_index, members in enumerate(group_members):
            total = np.zeros(scan.shape, dtype=np.float64)
            invalid = np.zeros(scan.shape, dtype=bool)
            for frame_index in members:
                frame = np.asarray(source_data[int(frame_index)], dtype=np.float64)
                frame_invalid = (
                    ~np.isfinite(frame) | (frame < 0) | (frame >= saturation)
                )
                invalid |= frame_invalid
                total += np.where(frame_invalid, 0.0, frame)
            total[invalid] = np.nan
            aggregated[group_index] = total

        offsets = np.zeros(len(group_members) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(members) for members in group_members])
        target.create_dataset("group_offsets", data=offsets)
        target.create_dataset(
            "source_frame_indices", data=np.concatenate(group_members)
        )
        target.attrs["aggregation"] = (
            "angle-group raw-count sum; I0 summed; any invalid constituent masks that pixel"
        )
        target.attrs["angle_tolerance_deg"] = angle_tolerance
        target.attrs["source_scan"] = str(scan.path)

    return replace(
        scan,
        detector_path=output_path,
        detector_dataset="data",
        angles=grouped_angles,
        i0=grouped_i0,
    ), group_members


def detector_layout(shape):
    if shape != (512, 1028):
        raise ValueError("Only the documented EIGER2 CdTe 500K layout is supported")
    boundaries = np.zeros(shape, dtype=bool)
    boundaries[:2] = boundaries[-2:] = True
    boundaries[:, :2] = boundaries[:, -2:] = True
    boundaries[254:258] = True
    for start, stop in ((254, 258), (511, 517), (770, 774)):
        boundaries[:, start:stop] = True
    sensors = np.full(shape, -1, dtype=np.int16)
    index = 0
    for start_column, stop_column in ((2, 254), (258, 511), (517, 770), (774, 1026)):
        for start_row, stop_row in ((2, 254), (258, 510)):
            sensors[start_row:stop_row, start_column:stop_column] = index
            index += 1
    return boundaries, sensors


def _geometry_from_config(config):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
        json.dump(config, file)
        file.flush()
        return Goniometer.sload(file.name)


def validate_azimuth_range(azimuth_range):
    """Normalize a directed, inclusive chi wedge in degrees; None keeps all chi.

    Bounds may cross the +/-180-degree seam, e.g. (170, -170). Equal
    endpoints are invalid; a +/-360-degree span explicitly selects all chi.
    """
    if azimuth_range is None:
        return None
    message = (
        "azimuth_range must be None or two finite degree bounds with 0 < width <= 360"
    )
    try:
        limits = np.asarray(azimuth_range, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError(message)
    lower, upper = map(float, limits)
    if not 0 < abs(upper - lower) <= 360:
        raise ValueError(message)
    return lower, upper


def azimuth_mask(model, angle, integrator=None):
    """True outside the model's chi wedge, evaluated at this motor position.

    This is a pixel-centre mask, not fractional clipping at wedge boundaries.
    It is geometry dependent and must never be folded into a fixed pixel mask.
    Models without an azimuth_range attribute retain legacy all-chi behaviour.
    """
    limits = validate_azimuth_range(getattr(model, "azimuth_range", None))
    if limits is None:
        return np.zeros(model.shape, dtype=bool)
    if integrator is None:
        integrator = model.geometry.get_ai(float(angle))
    chi = np.rad2deg(integrator.center_array(model.shape, unit="chi_rad"))
    lower, upper = limits
    if abs(upper - lower) == 360:
        return ~np.isfinite(chi)
    width = (upper - lower) % 360
    offset = (chi - lower) % 360
    # Include endpoints despite radian/degree round-off, including at the seam.
    inside = (offset <= width + 1e-10) | (offset >= 360 - 1e-10)
    return ~np.isfinite(chi) | ~inside


class PhysicalModel:
    def __init__(
        self, geometry_config, scan, polarization=0.95, density=6.2, azimuth_range=None
    ):
        self.azimuth_range = validate_azimuth_range(azimuth_range)
        self.config = geometry_config
        self.geometry = _geometry_from_config(geometry_config)
        if not np.isclose(
            self.geometry.wavelength, scan.wavelength_m, rtol=1e-6, atol=0
        ):
            raise ValueError("Scan and geometry wavelengths do not match")
        if int(geometry_config["detector_config"]["orientation"]) != scan.orientation:
            raise ValueError("Scan and geometry detector orientations do not match")
        if not -1 <= polarization <= 1 or density <= 0:
            raise ValueError("Invalid polarization factor or sensor density")
        self.shape = scan.shape
        self.polarization = polarization
        self.density = density
        self.energy = scan.energy_kev
        self.thickness = scan.thickness_m
        self.material = scan.material
        self.mu = xraylib.CS_Total_CP(self.material, self.energy) * density
        self.transmission = float(np.exp(-self.mu * self.thickness * 100))
        self.efficiency = 1 - self.transmission
        if not 0 < self.transmission < 1:
            raise ValueError("Degenerate attenuation model")

    def arrays(self, angle):
        integrator = self.geometry.get_ai(float(angle))
        theta = integrator.center_array(self.shape, unit="2th_rad")
        q_pixels = 4 * np.pi * np.sin(theta / 2) / (integrator.wavelength * 1e9)
        prefactor = integrator.pixel1 * integrator.pixel2 / integrator.dist**2
        absorption = (
            integrator.calc_transmission(self.transmission, shape=self.shape)
            * self.efficiency
        )
        norm = (
            integrator.solidAngleArray(self.shape)
            * prefactor
            * integrator.polarization(self.shape, factor=self.polarization)
            * absorption
        )
        return integrator, q_pixels, norm, absorption, prefactor

    def vertical_check(self):
        rows = []
        for angle in (20, 40, 60):
            integrator = self.geometry.get_ai(angle)
            ray_z, ray_y, ray_x = integrator.calc_pos_zyx(
                d1=np.array([255.5]), d2=np.array([513.5])
            )
            fraction = float(abs(ray_y[0]) / np.hypot(ray_x[0], ray_y[0]))
            chi = integrator.center_array(self.shape, unit="chi_rad")
            rows.append(
                {
                    "angle_deg": angle,
                    "vertical_fraction": fraction,
                    "chi_deg": float(np.rad2deg(chi)[256, 514]),
                }
            )
            if fraction < 0.99:
                raise ValueError(
                    "Geometry is not predominantly vertical: verify polarization convention before proceeding"
                )
        return rows


def integrate_frame(
    model,
    counts,
    angle,
    monitor,
    mask,
    divisor,
    npt,
    radial_range,
    unit="q_nm^-1",
    frame_gain=1.0,
    saturation=1e7,
):
    integrator, q_pixels, norm, absorption, prefactor = model.arrays(angle)
    counts = np.asarray(counts, dtype=float)
    bad = (
        mask
        | azimuth_mask(model, angle, integrator)
        | ~np.isfinite(counts)
        | (counts < 0)
        | (counts >= saturation)
    )
    if not np.any(~bad):
        raise ValueError("Frame has no unmasked pixels")
    if not np.isfinite(frame_gain) or frame_gain <= 0:
        raise ValueError("Invalid frame gain")
    return integrator.integrate1d(
        np.where(bad, 0, counts / divisor),
        npt,
        mask=bad,
        variance=np.where(bad, 0, counts / divisor**2),
        unit=unit,
        radial_range=radial_range,
        method=METHOD,
        normalization_factor=float(monitor) * prefactor * frame_gain,
        correctSolidAngle=True,
        polarization_factor=model.polarization,
        absorption=absorption,
    )


def constant_pattern_check(model, base_mask, q_range, npt):
    integrator, q_pixels, norm, absorption, prefactor = model.arrays(40)
    result = integrate_frame(
        model, 1e10 * norm, 40, 1, base_mask, np.ones(model.shape), npt, q_range
    )
    np.testing.assert_allclose(result.intensity[result.count > 0], 1e10, rtol=3e-6)


def build_cache(scan, model, base_mask, q_range=(200, 400), npt=800, saturation=1e7):
    if not 0 < q_range[0] < q_range[1] or not 16 <= npt <= 65535:
        raise ValueError("Invalid fitting q range or bin count")
    cache = {}
    step = (q_range[1] - q_range[0]) / npt
    with h5py.File(scan.detector_path, "r") as file:
        for index, angle in enumerate(scan.angles):
            integrator, q_pixels, norm, absorption, prefactor = model.arrays(angle)
            counts = file[scan.detector_dataset][index]
            valid = (
                ~(base_mask | azimuth_mask(model, angle, integrator))
                & np.isfinite(counts)
                & (counts < saturation)
            )
            valid &= (q_pixels >= q_range[0]) & (q_pixels < q_range[1])
            if valid.any():
                pixels = np.flatnonzero(valid).astype(np.int32)
                cache[index] = {
                    "pixels": pixels,
                    "bins": np.minimum(
                        np.floor((q_pixels.ravel()[pixels] - q_range[0]) / step),
                        npt - 1,
                    ).astype(np.uint16),
                    "counts": counts.ravel()[pixels].astype(np.float32),
                    "log_norm": np.log(norm.ravel()[pixels] * scan.i0[index]).astype(
                        np.float32
                    ),
                }
    if len(cache) < 4:
        raise ValueError(
            "Too few contributing scan positions; choose a supported fitting range or denser scan"
        )
    return cache


def fit_response(
    cache,
    indices,
    sensor_ids,
    frame_count,
    npt=800,
    iterations=10,
    fit_frame_gains=True,
    fractional_floor=0.003,
    reject_z=5.0,
    minimum_observations=8,
    shrink_weight=300.0,
    frame_prior_sigma=0.01,
):
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2 or any(int(index) not in cache for index in indices):
        raise ValueError("Invalid fit positions")
    shape = sensor_ids.shape
    sensors_flat = sensor_ids.ravel()
    pixel_log = np.zeros(sensor_ids.size)
    sensor_log = np.zeros(8)
    frame_log = np.zeros(frame_count)
    profile_log = np.zeros(npt)
    profile_sum = np.zeros(npt)
    profile_weight = np.zeros(npt)
    for index in indices:
        frame = cache[int(index)]
        observed = np.log(frame["counts"] + 0.5) - frame["log_norm"]
        weight = 1 / (1 / (frame["counts"] + 0.5) + fractional_floor**2)
        profile_sum += np.bincount(
            frame["bins"], weights=weight * observed, minlength=npt
        )
        profile_weight += np.bincount(frame["bins"], weights=weight, minlength=npt)
    supported = profile_weight > 0
    if supported.sum() < npt // 2:
        raise ValueError("Insufficient fitting-q support")
    profile_log[supported] = profile_sum[supported] / profile_weight[supported]
    profile_log = np.interp(
        np.arange(npt), np.flatnonzero(supported), profile_log[supported]
    )
    history = []
    for iteration in range(iterations):
        previous_pixel, previous_sensor, previous_frame = (
            pixel_log.copy(),
            sensor_log.copy(),
            frame_log.copy(),
        )
        weights_by_frame = {}
        profile_sum[:] = profile_weight[:] = 0
        for index in indices:
            index = int(index)
            frame = cache[index]
            pixels, bins = frame["pixels"], frame["bins"]
            sensors = sensors_flat[pixels]
            observed = np.log(frame["counts"] + 0.5) - frame["log_norm"]
            model = (
                profile_log[bins]
                + pixel_log[pixels]
                + sensor_log[sensors]
                + frame_log[index]
            )
            predicted = np.exp(np.clip(model + frame["log_norm"], -10, 30))
            scale = np.sqrt(1 / np.maximum(predicted, 1) + fractional_floor**2)
            z = (observed - model) / scale
            weights = 1 / scale**2 * np.minimum(1, 2 / np.maximum(np.abs(z), 1e-12))
            if iteration >= 2:
                weights[np.abs(z) > reject_z] = 0
            weights_by_frame[index] = weights
            residual = (
                observed - pixel_log[pixels] - sensor_log[sensors] - frame_log[index]
            )
            profile_sum += np.bincount(bins, weights=weights * residual, minlength=npt)
            profile_weight += np.bincount(bins, weights=weights, minlength=npt)
        supported = profile_weight > 0
        profile_log[supported] = profile_sum[supported] / profile_weight[supported]
        profile_log = np.interp(
            np.arange(npt), np.flatnonzero(supported), profile_log[supported]
        )
        pixel_sum, pixel_weight = np.zeros(sensor_ids.size), np.zeros(sensor_ids.size)
        observations = np.zeros(sensor_ids.size, dtype=np.uint32)
        for index in indices:
            index = int(index)
            frame = cache[index]
            pixels, bins = frame["pixels"], frame["bins"]
            weights = weights_by_frame[index]
            residual = (
                np.log(frame["counts"] + 0.5)
                - frame["log_norm"]
                - profile_log[bins]
                - sensor_log[sensors_flat[pixels]]
                - frame_log[index]
            )
            pixel_sum[pixels] += weights * residual
            pixel_weight[pixels] += weights
            observations[pixels] += weights > 0
        usable = observations >= minimum_observations
        pixel_log[:] = 0
        pixel_log[usable] = np.clip(
            pixel_sum[usable] / (pixel_weight[usable] + shrink_weight), -0.15, 0.15
        )
        for sensor_index in range(8):
            selected = usable & (sensors_flat == sensor_index)
            if not selected.any():
                raise ValueError(
                    f"Sensor {sensor_index + 1} has too few repeated pixel observations; lower minimum_observations only with justification or use a denser scan"
                )
            offset = np.average(pixel_log[selected], weights=pixel_weight[selected])
            pixel_log[selected] -= offset
            sensor_log[sensor_index] += offset
        sensor_sum, sensor_weight = np.zeros(8), np.zeros(8)
        for index in indices:
            index = int(index)
            frame = cache[index]
            pixels, bins = frame["pixels"], frame["bins"]
            weights = weights_by_frame[index]
            residual = (
                np.log(frame["counts"] + 0.5)
                - frame["log_norm"]
                - profile_log[bins]
                - pixel_log[pixels]
                - frame_log[index]
            )
            sensor_sum += np.bincount(
                sensors_flat[pixels], weights=weights * residual, minlength=8
            )
            sensor_weight += np.bincount(
                sensors_flat[pixels], weights=weights, minlength=8
            )
        sensor_log = np.clip(sensor_sum / (sensor_weight + 1 / 0.05**2), -0.10, 0.10)
        offset = sensor_log.mean()
        sensor_log -= offset
        profile_log += offset
        if fit_frame_gains:
            for index in indices:
                index = int(index)
                frame = cache[index]
                pixels, bins = frame["pixels"], frame["bins"]
                weights = weights_by_frame[index]
                residual = (
                    np.log(frame["counts"] + 0.5)
                    - frame["log_norm"]
                    - profile_log[bins]
                    - pixel_log[pixels]
                    - sensor_log[sensors_flat[pixels]]
                )
                frame_log[index] = np.clip(
                    np.sum(weights * residual)
                    / (weights.sum() + 1 / frame_prior_sigma**2),
                    -0.05,
                    0.05,
                )
            offset = frame_log[indices].mean()
            frame_log[indices] -= offset
            profile_log += offset
        rejected = sum(
            np.count_nonzero(weights == 0) for weights in weights_by_frame.values()
        )
        total = sum(len(weights) for weights in weights_by_frame.values())
        change = max(
            np.max(np.abs(pixel_log - previous_pixel)),
            np.max(np.abs(sensor_log - previous_sensor)),
            np.max(np.abs(frame_log - previous_frame)),
        )
        history.append(
            {
                "iteration": iteration + 1,
                "rejected_percent": 100 * rejected / total,
                "max_log_gain_change": float(change),
            }
        )
        print(history[-1])
    rejected = np.zeros(sensor_ids.size, dtype=np.uint32)
    observed_count = np.zeros(sensor_ids.size, dtype=np.uint32)
    for index in indices:
        frame = cache[int(index)]
        pixels, bins = frame["pixels"], frame["bins"]
        observed = np.log(frame["counts"] + 0.5) - frame["log_norm"]
        model = (
            profile_log[bins]
            + pixel_log[pixels]
            + sensor_log[sensors_flat[pixels]]
            + frame_log[index]
        )
        scale = np.sqrt(
            1 / np.maximum(np.exp(np.clip(model + frame["log_norm"], -10, 30)), 1)
            + fractional_floor**2
        )
        rejected[pixels] += np.abs((observed - model) / scale) > reject_z
        observed_count[pixels] += 1
    fraction = np.divide(
        rejected,
        observed_count,
        out=np.zeros(sensor_ids.size),
        where=observed_count > 0,
    )
    bad_pixels = (
        (observed_count >= minimum_observations) & (rejected >= 3) & (fraction >= 0.2)
    )
    sensor_map = np.ones(shape)
    for index in range(8):
        sensor_map[sensor_ids == index] = np.exp(sensor_log[index])
    pixel_map = np.exp(pixel_log).reshape(shape)
    divisor = pixel_map * sensor_map
    return {
        "pixel_divisor": pixel_map,
        "sensor_divisor": sensor_map,
        "sensor_gains": np.exp(sensor_log),
        "combined_divisor": divisor,
        "bad_pixel_mask": bad_pixels.reshape(shape),
        "frame_gains": np.exp(frame_log),
        "profile_log": profile_log,
        "observations": observed_count.reshape(shape),
        "outlier_fraction": fraction.reshape(shape),
        "history": history,
        "fit_indices": indices,
        "supported_q_bins": supported,
        "fit_frame_gains": fit_frame_gains,
    }


def merge_results(results):
    if not results:
        raise ValueError("No integrations to merge")
    axis = next(iter(results.values())).radial.copy()
    totals = np.zeros((4, len(axis)))
    for result in results.values():
        np.testing.assert_allclose(result.radial, axis)
        for field_index, field in enumerate(
            ("sum_signal", "sum_normalization", "sum_variance", "count")
        ):
            values = np.asarray(getattr(result, field), dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Nonfinite integration accumulator: {field}")
            totals[field_index] += values
    supported = totals[1] > 0
    intensity = np.divide(
        totals[0], totals[1], out=np.full(len(axis), np.nan), where=supported
    )
    sigma = np.divide(
        np.sqrt(totals[2]), totals[1], out=np.full(len(axis), np.nan), where=supported
    )
    return {
        "axis": axis,
        "intensity": intensity,
        "sigma": sigma,
        "accumulators": totals,
    }


def overlap_summary(results):
    centered, absolute = [], []
    indices = sorted(results)
    for left_index, right_index in zip(indices[:-1], indices[1:]):
        left, right = results[left_index], results[right_index]
        valid = (
            (left.count >= 100)
            & (right.count >= 100)
            & (left.intensity > 0)
            & (right.intensity > 0)
        )
        if valid.sum() >= 20:
            ratio = np.log(left.intensity[valid] / right.intensity[valid])
            centered.append(np.sqrt(np.mean((ratio - np.median(ratio)) ** 2)))
            absolute.append(np.sqrt(np.mean(ratio**2)))
    if not centered:
        raise ValueError("Too few overlapping bins for validation")
    return {
        "centered_overlap_percent": float(100 * np.median(centered)),
        "absolute_overlap_percent": float(100 * np.median(absolute)),
    }


def _position_accumulators(group):
    """Validate the common axis and four-field position accumulator layout."""
    try:
        frames = np.asarray(group["per_frame"], dtype=float)
        axis = np.asarray(group["axis"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("group requires numeric per_frame and axis arrays") from error
    if (
        frames.ndim != 3
        or frames.shape[1] != 4
        or frames.shape[0] == 0
        or frames.shape[2] == 0
    ):
        raise ValueError(
            "per_frame must have shape (positions, 4, bins), with nonempty positions/bins"
        )
    if axis.shape != (frames.shape[2],) or not np.all(np.isfinite(axis)):
        raise ValueError("axis must be finite and match per_frame bins")
    return frames


def refine_position_scales(
    group, *, minimum_pixel_weight=100, minimum_overlap_bins=20, fractional_floor=0.003
) -> dict:
    """Fit one positive intensity divisor per position from same-bin overlaps.

    ``group`` is a process_scan merged_2theta payload, integrated with UNITY
    frame gains. This is a required caller precondition: a group alone does
    not carry the processed-level gain provenance. No merged intensity, fit
    range restriction, smoothing, prior, bounds or Q-dependent fit is used.

    Each pair contributes one edge: the inverse-log-variance weighted mean
    of log(I_i)-log(I_j). Its bin variance is V_i/S_i**2 + V_j/S_j**2
    + fractional_floor**2, where S and V are the signal/variance accumulators.
    The floor is strictly positive so zero raw variance remains well-defined.
    Invalid/nonpositive intensities, negative/nonfinite variances, nonpositive
    normalizations and insufficient pixel weights are excluded per sample.
    Every position must belong to one connected overlap graph (at least two
    positions); otherwise ValueError is raised without fallback.

    Weighted graph least squares solves log(g_i)-log(g_j)=delta_ij, with
    equal-position mean log(g)=0, so I_i/g_i is on a common scale. Edges share
    observations and bins can be correlated: weights and residual RMS are
    descriptive, NOT calibrated standard errors or independent validation.
    Edge deltas, residuals and RMS diagnostics are in natural-log units;
    n_pair_bins counts bin uses across pairs, not independent observations.
    """
    for name, value in (
        ("minimum_pixel_weight", minimum_pixel_weight),
        ("fractional_floor", fractional_floor),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if (
        isinstance(minimum_overlap_bins, (bool, np.bool_))
        or not isinstance(minimum_overlap_bins, (int, np.integer))
        or minimum_overlap_bins < 1
    ):
        raise ValueError("minimum_overlap_bins must be a positive integer")
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        floor_variance = np.square(float(fractional_floor))
        inverse_floor = np.divide(1.0, floor_variance)
    if (
        not np.isfinite(floor_variance)
        or floor_variance <= 0
        or not np.isfinite(inverse_floor)
    ):
        raise ValueError("fractional_floor squared must have a finite positive inverse")

    frames = _position_accumulators(group)
    n_positions = frames.shape[0]
    if n_positions < 2:
        raise ValueError("Overlap refinement requires at least two connected positions")
    signal, normalization, variance, count = (frames[:, index] for index in range(4))
    valid = (
        np.all(np.isfinite(frames), axis=1)
        & (signal > 0)
        & (normalization > 0)
        & (variance >= 0)
        & (count >= minimum_pixel_weight)
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        intensity = signal / normalization
        relative_variance = (np.sqrt(variance) / signal) ** 2
    valid &= np.isfinite(intensity) & (intensity > 0) & np.isfinite(relative_variance)
    log_intensity = np.zeros_like(intensity)
    np.log(intensity, out=log_intensity, where=valid)
    edges = []
    neighbors = [set() for _ in range(n_positions)]
    for i in range(n_positions):
        for j in range(i + 1, n_positions):
            with np.errstate(
                over="ignore", under="ignore", divide="ignore", invalid="ignore"
            ):
                log_variance = (
                    relative_variance[i] + relative_variance[j] + floor_variance
                )
                bin_weight = 1.0 / log_variance
            selected = valid[i] & valid[j] & np.isfinite(bin_weight) & (bin_weight > 0)
            n_bins = int(selected.sum())
            if n_bins < minimum_overlap_bins:
                continue
            weights = bin_weight[selected]
            weight = float(weights.sum())
            if not np.isfinite(weight):
                raise ValueError("Nonfinite total overlap edge weight")
            delta = float(
                np.dot(
                    weights / weight,
                    log_intensity[i, selected] - log_intensity[j, selected],
                )
            )
            edges.append(
                {"i": i, "j": j, "n_bins": n_bins, "delta": delta, "weight": weight}
            )
            neighbors[i].add(j)
            neighbors[j].add(i)

    components = []
    unseen = set(range(n_positions))
    while unseen:
        pending = [min(unseen)]
        unseen.remove(pending[0])
        component = []
        while pending:
            position = pending.pop()
            component.append(position)
            discovered = neighbors[position] & unseen
            unseen.difference_update(discovered)
            pending.extend(sorted(discovered))
        components.append(sorted(component))
    if len(components) != 1:
        unsupported = [i for i in range(n_positions) if not neighbors[i]]
        raise ValueError(
            f"Disconnected overlap graph: components={components}; unsupported positions={unsupported}"
        )

    design = np.zeros((len(edges), n_positions))
    for row, edge in enumerate(edges):
        design[row, edge["i"]] = 1.0
        design[row, edge["j"]] = -1.0
    weights = np.array([edge["weight"] for edge in edges])
    delta = np.array([edge["delta"] for edge in edges])
    root_weight = np.sqrt(weights / weights.max())
    # Anchor the last position for the solve, then impose the equal-log gauge.
    solution, _, rank, _ = np.linalg.lstsq(
        design[:, :-1] * root_weight[:, None], delta * root_weight, rcond=None
    )
    if rank != n_positions - 1:
        raise ValueError("Overlap graph is numerically rank deficient")
    log_gains = np.r_[solution, 0.0]
    log_gains -= log_gains.mean()
    with np.errstate(over="ignore", under="ignore"):
        gains = np.exp(log_gains)
    if not np.all(np.isfinite(gains) & (gains > 0)):
        raise ValueError("Overlap solution does not yield finite positive gains")
    residual = delta - design @ log_gains
    for edge, after in zip(edges, residual):
        edge.update({"residual_before": edge["delta"], "residual_after": float(after)})
    normalized_weights = weights / weights.max()
    normalized_weights /= normalized_weights.sum()
    return {
        "frame_gains": gains,
        "edges": edges,
        "components": components,
        "n_components": len(components),
        "n_positions": n_positions,
        "n_edges": len(edges),
        "position_valid_bins": valid.sum(axis=1).tolist(),
        "position_edge_counts": [len(adjacent) for adjacent in neighbors],
        "n_pair_bins": sum(edge["n_bins"] for edge in edges),
        "edge_rms_before": float(np.sqrt(np.mean(delta**2))),
        "edge_rms_after": float(np.sqrt(np.mean(residual**2))),
        "weighted_edge_rms_before": float(
            np.sqrt(np.dot(normalized_weights, delta**2))
        ),
        "weighted_edge_rms_after": float(
            np.sqrt(np.dot(normalized_weights, residual**2))
        ),
        "settings": {
            "minimum_pixel_weight": float(minimum_pixel_weight),
            "minimum_overlap_bins": int(minimum_overlap_bins),
            "fractional_floor": float(fractional_floor),
        },
        "gauge": "equal-position mean log(frame_gains) = 0",
        "uncertainty": "Correlated overlap edges; no calibrated scale standard errors",
    }


def apply_position_scales(processed, gains) -> dict:
    """Return a new processed payload with normalization multiplied by gains.

    The original processed frame_gains must be exactly unity, not merely close.
    Signal, variance and pixel count are unchanged in every group, including
    merged_q and merged_2theta. Recompute merged I=S/N and sigma=sqrt(V)/N;
    sigma is conditional on the fitted scales and EXCLUDES their uncertainty.
    Preserve other metadata (including masks, axes and fit-range flags).
    Remove top-level/group 'results': underlying pyFAI objects would be stale.
    Input arrays/dictionaries are never written to; unchanged metadata may be
    shared with the returned payload.
    """
    try:
        gains = np.asarray(gains, dtype=float)
        original_gains = np.asarray(processed["frame_gains"], dtype=float)
        groups = processed["groups"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "processed requires groups and unity frame_gains; gains must be numeric"
        ) from error
    if (
        gains.ndim != 1
        or gains.size == 0
        or not np.all(np.isfinite(gains) & (gains > 0))
    ):
        raise ValueError("gains must be a nonempty finite positive 1D array")
    if original_gains.shape != gains.shape or not np.all(original_gains == 1.0):
        raise ValueError(
            "Original frame_gains must be exactly unity and match gains length (no double application)"
        )
    if not isinstance(groups, dict) or not groups:
        raise ValueError("processed groups must be a nonempty dict")
    updated_groups = {}
    for name, group in groups.items():
        frames = _position_accumulators(group)
        if frames.shape[0] != gains.size:
            raise ValueError(f"{name}: position count must match gains length")
        if not np.all(np.isfinite(frames)) or np.any(frames[:, 1:] < 0):
            raise ValueError(
                f"{name}: accumulators must be finite with nonnegative normalization, variance and count"
            )
        scaled = frames.copy()
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled[:, 1] *= gains[:, None]
            totals = scaled.sum(axis=0)
        if (
            not np.all(np.isfinite(totals))
            or not np.all(np.isfinite(scaled))
            or np.any((frames[:, 1] > 0) & (scaled[:, 1] <= 0))
        ):
            raise ValueError(
                f"{name}: scaled normalization/accumulators overflow or underflow"
            )
        supported = totals[1] > 0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            intensity = np.divide(
                totals[0],
                totals[1],
                out=np.full(totals.shape[1], np.nan),
                where=supported,
            )
            sigma = np.divide(
                np.sqrt(totals[2]),
                totals[1],
                out=np.full(totals.shape[1], np.nan),
                where=supported,
            )
        if not np.all(
            np.isfinite(intensity[supported]) & np.isfinite(sigma[supported])
        ):
            raise ValueError(
                f"{name}: scaled intensity/sigma must be finite in supported bins"
            )
        updated_groups[name] = {
            key: value for key, value in group.items() if key != "results"
        }
        updated_groups[name].update(
            per_frame=scaled, accumulators=totals, intensity=intensity, sigma=sigma
        )
    updated = {key: value for key, value in processed.items() if key != "results"}
    updated.update(groups=updated_groups, frame_gains=gains.copy())
    return updated


def validate_response(
    scan,
    model,
    cache,
    fit,
    test_indices,
    base_mask,
    q_range,
    npt,
    saturation=1e7,
    *,
    validate_frame_gains=True,
):
    """Compare responses, optionally inferring frame gains from lower-Q pixels.

    Disabling frame-gain validation removes the lower-Q support requirement
    and gain-corrected result; frame rows report unity gains. Higher-Q scoring
    uses the same cutoff and support requirements in either mode.
    """
    results = {"physical_only": {}, "detector_only": {}}
    if validate_frame_gains:
        results["lower_q_frame_gain"] = {}
    high_q_scores = {label: [] for label in results}
    frame_rows = []
    step = (q_range[1] - q_range[0]) / npt
    with h5py.File(scan.detector_path, "r") as file:
        for index in test_indices:
            index = int(index)
            frame = cache[index]
            bins = frame["bins"]
            cutoff = int(np.quantile(np.unique(bins), 0.4))
            inferred_gain = 1.0
            variants = [
                ("physical_only", np.ones(scan.shape), 1.0),
                ("detector_only", fit["combined_divisor"], 1.0),
            ]
            if validate_frame_gains:
                pixels = frame["pixels"]
                log_gain = np.log(fit["combined_divisor"].ravel()[pixels])
                residual = (
                    np.log(frame["counts"] + 0.5)
                    - frame["log_norm"]
                    - fit["profile_log"][bins]
                    - log_gain
                )
                selected = (bins <= cutoff) & ~fit["bad_pixel_mask"].ravel()[pixels]
                if selected.sum() < 100:
                    raise ValueError(
                        "Too few lower-q pixels for independent frame-gain validation"
                    )
                inferred_gain = float(
                    np.exp(np.clip(np.median(residual[selected]), -0.05, 0.05))
                )
                variants.append(
                    ("lower_q_frame_gain", fit["combined_divisor"], inferred_gain)
                )
            counts = file[scan.detector_dataset][index]
            for label, divisor, gain in variants:
                result = integrate_frame(
                    model,
                    counts,
                    scan.angles[index],
                    scan.i0[index],
                    base_mask | fit["bad_pixel_mask"],
                    divisor,
                    npt,
                    q_range,
                    frame_gain=gain,
                    saturation=saturation,
                )
                results[label][index] = result
                high = (
                    (np.arange(npt) > cutoff + 2)
                    & (result.count >= 100)
                    & (result.intensity > 0)
                    & fit["supported_q_bins"]
                )
                if not high.any():
                    raise ValueError("No supported higher-q bins for frame validation")
                ratio = np.log(result.intensity[high]) - fit["profile_log"][high]
                high_q_scores[label].append(100 * np.sqrt(np.mean(ratio**2)))
            frame_rows.append(
                {
                    "index": index,
                    "angle_deg": float(scan.angles[index]),
                    "inferred_gain": inferred_gain,
                    "cutoff_q_nm": q_range[0] + (cutoff + 0.5) * step,
                }
            )
    summary = [
        {
            "label": label,
            **overlap_summary(values),
            "higher_q_rms_percent": float(np.median(high_q_scores[label])),
        }
        for label, values in results.items()
    ]
    return {"summary": summary, "frame_rows": frame_rows, "results": results}


def _tag_nxdata(group, signal_name, axes, interpretation):
    """Modern NeXus plot declarations plus legacy dataset tags for DAWN readers."""
    group.attrs["NX_class"] = "NXdata"
    group.attrs["signal"] = signal_name
    group.attrs["axes"] = np.asarray(axes, dtype=h5py.string_dtype("utf-8"))
    signal = group[signal_name]
    signal.attrs["signal"] = 1
    signal.attrs["axes"] = ":".join(axes)
    signal.attrs["interpretation"] = interpretation
    for dimension, name in enumerate(axes):
        group.attrs[f"{name}_indices"] = dimension
        group[name].attrs["axis"] = dimension + 1
        group[name].attrs["primary"] = 1


def _tag_calibration_plot(file):
    """Link an NXentry default view to existing corrections without copying data."""
    correction = file["corrections"]
    for dimension, name, label in (
        (0, "detector_row", "Detector row (pixel index)"),
        (1, "detector_column", "Detector column (pixel index)"),
    ):
        if name not in correction:
            correction.create_dataset(
                name, data=np.arange(correction["combined_divisor"].shape[dimension])
            )
        correction[name].attrs["units"] = "1"
        correction[name].attrs["long_name"] = label
    _tag_nxdata(
        correction, "combined_divisor", ("detector_row", "detector_column"), "image"
    )
    correction["combined_divisor"].attrs["units"] = "1"
    correction["combined_divisor"].attrs["long_name"] = (
        "Combined pixel x sensor intensity divisor"
    )
    file.attrs["default"] = "entry"
    entry = file.require_group("entry")
    entry.attrs["NX_class"] = "NXentry"
    entry.attrs["default"] = "corrections"
    if "corrections" not in entry:
        entry["corrections"] = correction  # HDF5 hard link, not a second map.


def _tag_pattern_plot(file):
    """Select intensity versus scattering coordinate and label conditional sigma."""
    file.attrs["default"] = "entry"
    entry = file["entry"]
    entry.attrs["NX_class"] = "NXentry"
    entry.attrs["default"] = "merged_2theta"
    for name in ("merged_2theta", "merged_q"):
        if name not in entry:
            continue
        group = entry[name]
        axis_name = "q_nm" if name == "merged_q" else "two_theta_deg"
        _tag_nxdata(group, "intensity", (axis_name,), "spectrum")
        group[axis_name].attrs["units"] = "nm^-1" if name == "merged_q" else "degrees"
        group[axis_name].attrs["long_name"] = (
            "Q (nm^-1)" if name == "merged_q" else "Scattering 2theta (degrees)"
        )
        group["intensity"].attrs["long_name"] = "Normalized intensity (a.u.)"
        group["sigma"].attrs["long_name"] = "Conditional Poisson sigma (a.u.)"
        group["sigma"].attrs["uncertainty"] = UNCERTAINTY
        group["intensity"].attrs["uncertainties"] = "sigma"
        if "intensity_errors" not in group:
            group["intensity_errors"] = group["sigma"]


def save_calibration(
    path, scan, model, fit, base_mask, sensor_ids, fit_settings, validation
):
    # Persist the actual model selection even for callers other than the notebook.
    fit_settings = {
        **fit_settings,
        "azimuth_range_deg": validate_azimuth_range(
            getattr(model, "azimuth_range", None)
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as file:
        file.attrs["schema"] = SCHEMA
        file.attrs["status"] = (
            "exploratory; validate on independent scans before production use"
        )
        file.attrs["source_scan"] = str(scan.path)
        file.attrs["pyfai_version"] = pyFAI.version
        file.attrs["fit_settings_json"] = json.dumps(fit_settings)
        file.create_dataset("geometry_json", data=json.dumps(model.config))
        physical = file.create_group("physical")
        for name, value in {
            "polarization_factor": model.polarization,
            "density_g_cm3": model.density,
            "energy_kev": model.energy,
            "sensor_thickness_m": model.thickness,
            "material": model.material,
            "orientation": scan.orientation,
            "wavelength_m": scan.wavelength_m,
            "linear_attenuation_cm_inverse": model.mu,
            "normal_transmission": model.transmission,
            "normal_efficiency": model.efficiency,
        }.items():
            physical.attrs[name] = value
        physical.attrs["model"] = (
            "absolute solid angle, pyFAI polarization, xraylib planar Beer-Lambert CdTe absorption; not measured detector QE"
        )
        correction = file.create_group("corrections")
        for name in (
            "pixel_divisor",
            "sensor_divisor",
            "sensor_gains",
            "combined_divisor",
            "bad_pixel_mask",
            "observations",
            "outlier_fraction",
        ):
            correction.create_dataset(
                name, data=fit[name], compression="gzip", shuffle=True
            )
        correction.create_dataset("base_mask", data=base_mask, compression="gzip")
        correction.create_dataset("sensor_ids", data=sensor_ids, compression="gzip")
        correction.attrs["semantics"] = (
            "Divide raw counts by combined_divisor, raw variance by its square"
        )
        diagnostics = file.create_group("fit_diagnostics")
        diagnostics.attrs["history_json"] = json.dumps(fit["history"])
        diagnostics.create_dataset("frame_angles_deg", data=scan.angles)
        diagnostics.create_dataset("frame_i0", data=scan.i0)
        diagnostics.create_dataset("frame_gains", data=fit["frame_gains"])
        diagnostics.create_dataset("fit_indices", data=fit["fit_indices"])
        diagnostics.create_dataset("profile_log", data=fit["profile_log"])
        diagnostics.create_dataset("supported_q_bins", data=fit["supported_q_bins"])
        diagnostics.attrs["frame_gains_not_transferable"] = True
        diagnostics.attrs["validation_json"] = json.dumps(validation)
        _tag_calibration_plot(file)
    loaded = load_calibration(path)
    for name in (
        "combined_divisor",
        "pixel_divisor",
        "sensor_divisor",
        "bad_pixel_mask",
    ):
        np.testing.assert_array_equal(loaded[name], fit[name])
    return path.resolve()


def load_calibration(path):
    path = Path(path).resolve()
    with h5py.File(path, "r") as file:
        if file.attrs.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported calibration schema; expected {SCHEMA}")
        correction = file["corrections"]
        payload = {
            name: _array(correction, name)
            for name in (
                "combined_divisor",
                "pixel_divisor",
                "sensor_divisor",
                "sensor_gains",
                "bad_pixel_mask",
                "base_mask",
                "sensor_ids",
            )
        }
        payload.update(
            {
                "path": path,
                "geometry": json.loads(_text(file["geometry_json"][()])),
                "physical": dict(file["physical"].attrs),
                "fit_settings": json.loads(file.attrs["fit_settings_json"]),
                "source_scan": _text(file.attrs["source_scan"]),
                "frame_angles": _array(file["fit_diagnostics"], "frame_angles_deg"),
                "frame_i0": _array(file["fit_diagnostics"], "frame_i0"),
                "frame_gains": _array(file["fit_diagnostics"], "frame_gains"),
            }
        )
    divisor = payload["combined_divisor"]
    if divisor.shape != (512, 1028) or not np.all(np.isfinite(divisor) & (divisor > 0)):
        raise ValueError("Invalid saved detector divisor")
    np.testing.assert_allclose(
        payload["pixel_divisor"] * payload["sensor_divisor"], divisor
    )
    return payload


def model_for_processing(scan, calibration):
    physical = calibration["physical"]
    for name, actual in (
        ("energy_kev", scan.energy_kev),
        ("sensor_thickness_m", scan.thickness_m),
        ("wavelength_m", scan.wavelength_m),
    ):
        if not np.isclose(actual, physical[name], rtol=1e-6, atol=0):
            raise ValueError(
                f"Scan/calibration {name} mismatch; do not silently transfer this calibration"
            )
    if (
        scan.material != _text(physical["material"])
        or scan.orientation != physical["orientation"]
    ):
        raise ValueError("Scan/calibration material or detector orientation mismatch")
    model = PhysicalModel(
        calibration["geometry"],
        scan,
        float(physical["polarization_factor"]),
        float(physical["density_g_cm3"]),
        azimuth_range=calibration["fit_settings"].get("azimuth_range_deg"),
    )
    np.testing.assert_allclose(
        model.transmission, physical["normal_transmission"], rtol=1e-6
    )
    model.vertical_check()
    return model


def theta_range_for_scan(scan, model, mask, theta_max=None):
    """Return (0, maximum) 2theta degrees without reading detector counts.

    The automatic maximum uses finite, unmasked pyFAI pixel centres over all
    scan positions, including the model's per-position azimuth wedge, not
    pixel corners or count-dependent frame masks.
    """
    message = (
        "theta_max must be None or a finite number with 0 < theta_max <= 180 degrees"
    )
    if theta_max is not None:
        if not isinstance(theta_max, (int, float, np.integer, np.floating)):
            raise ValueError(message)
        upper = float(theta_max)
        if not np.isfinite(upper) or not 0 < upper <= 180:
            raise ValueError(message)
        return 0.0, upper
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != scan.shape:
        raise ValueError("Theta-range mask must match the detector shape")
    if np.all(mask):
        raise ValueError("Cannot determine theta range: no unmasked pixels")
    upper = -np.inf
    for angle in scan.angles:
        integrator = model.geometry.get_ai(float(angle))
        theta = np.rad2deg(integrator.center_array(scan.shape, unit="2th_rad"))
        valid = ~(mask | azimuth_mask(model, angle, integrator)) & np.isfinite(theta)
        if np.any(valid):
            upper = max(upper, float(theta[valid].max()))
    if not np.isfinite(upper) or not 0 < upper <= 180:
        raise ValueError(
            "Cannot determine theta range: no finite, positive unmasked pyFAI maximum"
        )
    return 0.0, upper


def process_scan(
    scan,
    calibration,
    model,
    q_range=(200, 400),
    q_npt=800,
    theta_npt=1000,
    apply_calibration_frame_gains=False,
    allow_extrapolation=True,
    saturation=1e7,
    theta_range=None,
):
    """Integrate with optional explicit (lower, upper) 2theta limits in degrees.

    None starts at zero and uses the maximum unmasked pyFAI pixel centre.
    Explicit (lower, upper) tuples remain supported for existing callers.
    """
    if theta_range is not None:
        message = "theta_range must be a finite (lower, upper) tuple with 0 <= lower < upper <= 180 degrees"
        if not isinstance(theta_range, tuple) or len(theta_range) != 2:
            raise ValueError(message)
        try:
            limits = np.asarray(theta_range, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(message) from error
        if (
            limits.shape != (2,)
            or not np.all(np.isfinite(limits))
            or not 0 <= limits[0] < limits[1] <= 180
        ):
            raise ValueError(message)
        lower, upper = map(float, limits)
    fit_range = calibration["fit_settings"]["q_range"]
    if not allow_extrapolation:
        raise ValueError(
            "Full-range 2theta export extrapolates the fitted detector response; explicitly enable allow_extrapolation"
        )
    gains = np.ones(len(scan.angles))
    if apply_calibration_frame_gains:
        if str(scan.path) != calibration["source_scan"]:
            raise ValueError(
                "Calibration-frame gains may only be applied to the exact source scan"
            )
        np.testing.assert_array_equal(scan.angles, calibration["frame_angles"])
        np.testing.assert_array_equal(scan.i0, calibration["frame_i0"])
        gains = calibration["frame_gains"].copy()
    divisor = calibration["combined_divisor"]
    mask = (
        calibration["base_mask"].astype(bool)
        | calibration["bad_pixel_mask"].astype(bool)
        | scan.native_mask
    )
    if theta_range is None:
        lower, upper = theta_range_for_scan(scan, model, mask)
    results = {"merged_q": {}, "merged_2theta": {}}
    with h5py.File(scan.detector_path, "r") as file:
        for index, angle in enumerate(scan.angles):
            counts = file[scan.detector_dataset][index]
            for name, npt, radial_range, unit in (
                ("merged_q", q_npt, q_range, "q_nm^-1"),
                ("merged_2theta", theta_npt, (lower, upper), "2th_deg"),
            ):
                results[name][index] = integrate_frame(
                    model,
                    counts,
                    angle,
                    scan.i0[index],
                    mask,
                    divisor,
                    npt,
                    radial_range,
                    unit,
                    gains[index],
                    saturation,
                )
    payload = {
        name: {
            **merge_results(values),
            "per_frame": np.array(
                [
                    [
                        getattr(result, field)
                        for field in (
                            "sum_signal",
                            "sum_normalization",
                            "sum_variance",
                            "count",
                        )
                    ]
                    for result in values.values()
                ]
            ),
        }
        for name, values in results.items()
    }
    for name, group in payload.items():
        np.testing.assert_allclose(
            group["per_frame"].sum(axis=0), group["accumulators"]
        )
        q_values = (
            group["axis"]
            if name == "merged_q"
            else 4
            * np.pi
            * np.sin(np.deg2rad(group["axis"]) / 2)
            / (model.geometry.wavelength * 1e9)
        )
        group["within_fit_range"] = (q_values >= fit_range[0]) & (
            q_values <= fit_range[1]
        )
    return {
        "groups": payload,
        "frame_gains": gains,
        "mask": mask,
        "saturation": saturation,
        "results": results,
        "azimuth_range_deg": validate_azimuth_range(
            getattr(model, "azimuth_range", None)
        ),
    }


def save_processed(path, scan, calibration, model, processed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as file:
        file.attrs["schema"] = "eiger_processed_v1"
        file.attrs["calibration_path"] = str(calibration["path"])
        file.attrs["source_scan"] = str(scan.path)
        file.attrs["uncertainty"] = UNCERTAINTY
        file.attrs["no_response_refinement"] = True
        entry = file.create_group("entry")
        entry.attrs["default"] = "merged_2theta"
        for name, payload in processed["groups"].items():
            group = entry.create_group(name)
            axis_name = "q_nm" if name == "merged_q" else "two_theta_deg"
            group.attrs["unit"] = "q_nm^-1" if name == "merged_q" else "2th_deg"
            group.attrs["n_frames"] = len(scan.angles)
            group.attrs["reference_based_dynamic_rejection"] = False
            for key, values in (
                (axis_name, payload["axis"]),
                ("intensity", payload["intensity"]),
                ("sigma", payload["sigma"]),
                ("within_detector_fit_q_range", payload["within_fit_range"]),
                ("frame_accumulators", payload["per_frame"]),
            ):
                group.create_dataset(key, data=values, compression="gzip", shuffle=True)
            for index, field in enumerate(
                ("sum_signal", "sum_normalization", "sum_variance", "count")
            ):
                group.create_dataset(field, data=payload["accumulators"][index])
            group["frame_accumulators"].attrs["field_order"] = (
                "sum_signal,sum_normalization,sum_variance,count"
            )
            for key, values in (
                ("frame_angles_deg", scan.angles),
                ("frame_i0", scan.i0),
                ("frame_intensity_gains", processed["frame_gains"]),
                ("frame_effective_i0", scan.i0 * processed["frame_gains"]),
            ):
                group.create_dataset(key, data=values)
        correction = entry.create_group("calibration")
        correction.create_dataset(
            "geometry_json", data=json.dumps(calibration["geometry"])
        )
        for name in (
            "pixel_divisor",
            "sensor_divisor",
            "sensor_gains",
            "combined_divisor",
            "bad_pixel_mask",
        ):
            correction.create_dataset(name, data=calibration[name], compression="gzip")
        correction.create_dataset(
            "applied_fixed_mask", data=processed["mask"], compression="gzip"
        )
        correction.attrs["azimuth_range_deg_json"] = json.dumps(
            processed.get("azimuth_range_deg")
        )
        correction.attrs["azimuth_mask_convention"] = (
            "pyFAI chi degrees; inclusive pixel-centre wedge evaluated per motor position"
        )
        correction.attrs["source_calibration_scan"] = calibration["source_scan"]
        physical = entry.create_group("physical_corrections")
        for name, value in calibration["physical"].items():
            physical.attrs[name] = value
        physical.attrs["absolute_solid_angle"] = True
        entry.attrs["saturation_threshold"] = processed["saturation"]
        _tag_pattern_plot(file)
    with h5py.File(path, "r") as file:
        for name, payload in processed["groups"].items():
            group = file[f"entry/{name}"]
            np.testing.assert_allclose(
                group["intensity"][()], payload["intensity"], equal_nan=True
            )
            np.testing.assert_allclose(
                group["sigma"][()], payload["sigma"], equal_nan=True
            )
            totals = np.array(
                [
                    group[field][()]
                    for field in (
                        "sum_signal",
                        "sum_normalization",
                        "sum_variance",
                        "count",
                    )
                ]
            )
            np.testing.assert_allclose(
                group["frame_accumulators"][()].sum(axis=0), totals
            )
            valid = totals[1] > 0
            np.testing.assert_allclose(
                group["intensity"][()][valid], totals[0, valid] / totals[1, valid]
            )
            np.testing.assert_allclose(
                group["sigma"][()][valid], np.sqrt(totals[2, valid]) / totals[1, valid]
            )
            np.testing.assert_array_equal(
                group["frame_intensity_gains"][()], processed["frame_gains"]
            )
        np.testing.assert_array_equal(
            file["entry/calibration/combined_divisor"][()],
            calibration["combined_divisor"],
        )
    return path.resolve()
