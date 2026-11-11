# Local Input Data

The CLI accepts a scan path and, optionally, a geometry JSON override. Without
`--geometry`, it decodes the JSON blob at `/entry/plan_metadata/geometry_calibration`
in the NeXus scan directly. An explicit file takes precedence. Inputs can remain
on read-only beamline storage; the CLI never writes beside the source scan.

For a NeXus stem NAME, companion paths default to NAME-fastcs_eiger_000001.h5
(detector dataset `data`), NAME-i0.h5 (monitor dataset `entry/data/data`), and
NAME-fastcs_eiger_meta.h5 (detector mask/material/energy/thickness). If the default
detector file is absent the loader uses the linked NeXus detector dataset.
The CLI supports explicit companion-path overrides.

The bundled geometry is the October 15, 2026 goniometer calibration used for the
scan-100605 smoke test. Geometry must match the scan's wavelength and orientation;
do not treat that example as valid for every acquisition.

Raw detector and NeXus data are intentionally excluded from Git. Generated data
belong in a new run directory, normally under outputs, not here.
