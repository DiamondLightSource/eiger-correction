[![CI](https://github.com/DiamondLightSource/eiger-correction/actions/workflows/ci.yml/badge.svg)](https://github.com/DiamondLightSource/eiger-correction/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/DiamondLightSource/eiger-correction/branch/main/graph/badge.svg)](https://codecov.io/gh/DiamondLightSource/eiger-correction)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# eiger-correction

What            | Where
:---:           | :---:
Source          | <https://github.com/DiamondLightSource/eiger-correction>
Install         | `pip install git+https://github.com/DiamondLightSource/eiger-correction.git`
Docker          | `docker run ghcr.io/diamondlightsource/eiger-correction:latest`
Releases        | <https://github.com/DiamondLightSource/eiger-correction/releases>

A headless workflow that corrects polarisation and detector-response artefacts in
EIGER2 CdTe 500K scans. The `eiger-correction` command is defined in
[\_\_main\_\_.py](src/eiger_correction/__main__.py). The extraction stages are in
[extract_eiger_corrections.py](src/eiger_correction/extract_eiger_corrections.py),
scientific routines in [eiger_workflow.py](src/eiger_correction/eiger_workflow.py),
JPEG generation in [eiger_diagnostics.py](src/eiger_correction/eiger_diagnostics.py)
and the Markdown report in [eiger_report.py](src/eiger_correction/eiger_report.py).
No Jupyter installation is needed.

## Quick start

```bash
eiger-correction /dls/i15-1/data/2026/cm44163-4/i15-1-100605.nxs --output-dir /some/output/directory
```

This writes the following to the output directory:

```text
detector_corrections.h5   # detector calibration
pattern.h5                # final integrated pattern
pattern_2theta.xy         # two-column 2theta / intensity
angle_grouping.jpg
azimuth_geometry.jpg
detector_response.jpg
final_xy.jpg
overlap_final_abcd.jpg
position_gains.jpg
sampling_q.jpg
report.json               # machine-readable run record
report.md                 # human-readable report with embedded JPEGs
run.log                   # full process log
```

See [Outputs and unattended operation](#outputs-and-unattended-operation) for details.

## Interactive notebook

[notebooks/extract_eiger_corrections.ipynb](notebooks/extract_eiger_corrections.ipynb)
runs the same two-stage extraction step by step with the package functions, showing
every intermediate printout and plot inline. Settings live in the cell tagged
`parameters`, so it can also be run headless with papermill:

```bash
papermill notebooks/extract_eiger_corrections.ipynb run.ipynb \
    -p SCAN_PATH /dls/i15-1/data/2026/cm44163-4/i15-1-100605.nxs \
    -p OUTPUT_DIRECTORY /some/output/directory
```

It writes the calibration HDF5 and 2theta XY to `OUTPUT_DIRECTORY` (default
`./outputs`), and `run.ipynb` keeps all the figures.

## Environment and invocation

Python 3.11 or newer is required. Install the package and its dependencies with
`uv sync` (or `pip install .`). The dev container does this automatically. Then run
`eiger-correction SCAN` (or `python -m eiger_correction SCAN`) with the source
NeXus path.
By default, the goniometer calibration is decoded directly from the JSON blob at
`/entry/plan_metadata/geometry_calibration` in that scan. An explicit
`--geometry GEOMETRY_JSON` **overrides** the embedded geometry; an unreadable or
invalid override is an error, never a silent fallback. Missing/invalid embedded
JSON requires an explicit file. Geometry compatibility checks still apply.
The selected source and checksum are recorded in the JSON report and the source
is shown in the Markdown overview; exported HDF5 retains the selected geometry.
`--help` describes every option and `--version` prints the installed version.
`--output-dir DIRECTORY` chooses a **new** run directory; otherwise a unique UTC
timestamped directory is created under outputs. Existing directories are rejected,
even if empty: outputs are never overwritten and simultaneous jobs cannot collide.

For example, the real-data smoke test used scan 100605 from visit cm44163-4,
the bundled October 15 geometry
([src/eiger_correction/data/](src/eiger_correction/data/)), and `--raw-audit`.
All source inputs are opened read-only. Companion detector (`STEM-fastcs_eiger_000001.h5`),
I0 (`STEM-i0.h5`) and metadata (`STEM-fastcs_eiger_meta.h5`) filenames are
inferred from the NeXus stem; `--detector`, `--monitor` and `--metadata` override
them. If the detector file is absent, frames are read from
`entry/fastcs_eiger/data` in the NeXus file itself.

### Defaults matching the latest extraction notebook

| Setting | Default | CLI control |
|---|---|---|
| Azimuth | Single inclusive pyFAI chi wedge 67.5–112.5° | `--azimuth-range LOW HIGH`; `--all-azimuths` disables |
| Detector-fit Q interval | 10–40 Å⁻¹ | `--fit-q-range LOW HIGH` |
| Fit bins / iterations | 1200 / 10 | `--fit-bins`, `--iterations` |
| Final 2theta grid | 2001 bins, 0–80° | `--theta-bins`, `--theta-max` |
| Angle grouping tolerance | 0.001° | `--angle-tolerance`; zero means exact repeats |
| Raw saturation threshold | 10⁷ | `--saturation` |
| Overlap support | 100 pixel weight, 20 shared bins | `--overlap-min-pixels`, `--overlap-min-bins` |
| Overlap fractional floor | 0.003 | `--overlap-floor` |
| Diagnostic scattering interval | 30–80° | `--diagnostic-range LOW HIGH` |
| Independent validation | One held-out detector fold | `--validation basic\|holdout\|full` |
| Polarization factor | 0.95 | `--polarization` |
| CdTe density | 6.2 g/cm³ | `--density` |
| JPEG resolution | 140 dpi | `--jpeg-dpi` |

`--theta-max 0` selects the maximum finite unmasked pyFAI pixel-centre angle
across the scan, including the per-position azimuth selection. Narrower ranges or
wedges can disconnect the overlap graph: this is a hard failure, with no silent
threshold relaxation or fallback. Chi bounds are inclusive; seam-crossing bounds
such as 170, −170 are supported. The opposite wedge is not implicitly included.

## Outputs and unattended operation

Both HDF5 exports include NeXus `NXentry`/`NXdata`, root/entry `default`,
`signal` and `axes` tags for automatic plotting in readers such as DAWN.
The pattern defaults to intensity versus 2theta (degrees), with conditional
Poisson sigma linked as its uncertainties. The corrections file defaults to
the combined pixel × sensor divisor image, with detector column/row pixel-index
coordinates. Legacy dataset-level plot tags are included for older readers.
Existing dataset paths remain valid; additional hard links do not copy arrays.

Every run directory contains:

- **Calibration HDF5** (`detector_corrections.h5`): detector/pixel/sensor divisors, fixed masks, geometry,
	physical model, fit history and settings, held-out results, final source-only
	position gains, and separately labelled stage-1 nuisance gains.
- **Pattern HDF5** (`pattern.h5`): final 2theta axis, intensity, conditional Poisson sigma,
	signal/normalization/variance/pixel-weight accumulators per position and merged,
	fit-range flags, angles, I0, gains and applied corrections. There is deliberately
	no redundant Q integration.
- **Two-column XY** (`pattern_2theta.xy`): 2theta degrees and intensity, ten significant digits;
	unsupported bins are omitted, not filled or interpolated.
- **Seven JPEGs:** actual angle grouping; geometry-only azimuth masks; response
	maps/sensor gains/monitors; Q sampling; stage-1/final gains; full/zoom intensity;
	final overlap panels A–B **after final applied gains**, using the same final
	integration as the exported pattern. Unsupported intervals are recorded as skips
	and warnings rather than invalidating an otherwise supported extraction.
	The overlap figure notes that individual curves may spike where few pixels
	contribute; A retains all supported bins, while B uses the diagnostic
	pixel-weight threshold. The historical JPEG filename is retained for existing links.
- **JSON run report** (`report.json`): status, inputs, software versions and source/geometry
	hashes, settings, per-stage timing, validation metrics, fit history, all gains,
	graph edges/support, residual summaries, output names, warnings and exceptions.
	It is atomically checkpointed before/after each stage, not only on success.
- **Markdown run report** (`report.md`): a human-readable companion to the same
	JSON snapshot, including a run overview, warnings/failures, validation and timing
	tables, processing settings, product links and an embedded, labelled JPEG gallery.
	Open the Markdown preview in VS Code to see the images; the relative links
	remain valid when the complete run directory is moved. Both reports are
	updated at every checkpoint, including failed and interrupted runs. Detailed
	arrays and graph edges remain in JSON rather than overwhelming the overview.
- **Text run log** (`run.log`): Python stdout/stderr, fit progress, logging and failure
	tracebacks. Python warnings also appear in the report. No successful processing
	step depends on reading terminal stdout.

The report status is `running`, `completed`, `completed_with_warnings`, `failed`,
or `interrupted`. Exit status is zero for completed runs (including warnings),
nonzero for failure/interruption. A non-improving held-out metric is a warning,
not an invented acceptance threshold: automation should inspect the report's
warnings as well as the exit code. Argument-parser errors and inability to create
the run directory necessarily occur before a report can be created. A forced
kill/power loss leaves the last checkpoint marked running; it cannot finalize a
report or guarantee scratch cleanup. Normal exceptions and keyboard interruption
retain the traceback and clean up temporary grouped detector data. Failed runs
can contain partial products; only accept a completed report.

## Method and speed choices

1. Stream raw counts once into summed angle groups. Groups are anchored at their
	 minimum motor reading (not chained). Geometry uses the arithmetic mean of
	 motor readings, one equal contribution per raw frame, **never I0-weighted**.
	 I0 is summed. Any nonfinite/negative/saturated member rejects that group pixel.
2. Verify geometry/physical normalization, then build a sparse fit-only cache.
	 Stage 1 estimates a common log-Q profile, pixel/sensor response and nuisance
	 position scales, using the existing robust/shrunk fit unchanged.
3. Freeze the detector map and integrate each group **once in 2theta at unity
	 position scales**. The same read supplies integer pixel-centre sampling counts.
	 These counts are not photons or pyFAI's fractional pixel-splitting weights.
4. Solve one positive scalar per position jointly from same-bin full-range
	 overlaps. The graph must connect every position; the equal-position geometric
	 mean scale is one. Final scales **replace**, not multiply, nuisance scales.
	 Apply them to normalization accumulators only and rebuild the merge.
5. Export, read back/check outputs, and render JPEGs from existing results.

Removed notebook-only work: the low-angle raw-image mask preview, full raw-data
reread for sampling plots, separate-versus-summed exploratory integrations,
ungrouped benchmark, duplicate grouping preview algorithm, and duplicate Q
integration. Residual diagnostics are reused rather than recomputed for the stack.
One raw source pass is necessary for aggregation; later fitting/validation and
integration read the much smaller temporary grouped file. `--raw-audit` explicitly
adds a small independent source reread. No smoothing, tapering, background
subtraction, additive offset, Q-dependent scale or new fit model was introduced.

## Validation strategy

**Every run (`--validation basic` or stronger):** check input shape/metadata,
positive I0, complete grouping/mean-angle/summed-I0 identities, vertical geometry,
constant-pattern physical normalization, fit support, connected overlap graph,
positive finite gains, mean-log gauge, invariant signal/variance/pixel counts,
correct normalization-only scale application, accumulator conservation, and
intensity/sigma identities. Calibration and pattern HDF5 are reopened and checked;
XY is read back against the integrated data. Unsupported bins are reported.

**Default `--validation holdout`:** one additional detector fit trains on alternating
contributing groups, holding the others out entirely. Compare physical-only versus
detector-corrected held-out overlap (absolute and centered) and higher-Q RMS.
No gain is inferred on the test data, avoiding that extra diagnostic fit/integration.
Half-data fits use four observations per pixel, versus eight for the full fit;
this never weakens the full-fit support requirement.

**`--validation full`:** reverse the train/test roles for a second fold, retain
the notebook's lower-Q-inferred gain / higher-Q check, and report fold-to-fold
sensor log-gain RMS. This costs two extra fits rather than the default one.

**`--raw-audit`:** independently reconstruct the highest-position group from raw
frames, reject invalid constituents and reintegrate it with the final gain. Compare
all four accumulators against scalar application (3 × 10⁻⁶ relative tolerance).
Useful after geometry, masking or integration-code changes; normally unnecessary
for every scan after acceptance testing.

**Interpretation:** held-out tests probe detector generalization within this scan,
not transfer to another scan. Stage-2 overlap residuals are
**in-sample descriptive diagnostics**, not independent validation or calibrated
uncertainties. Recommended further acceptance testing is independent scans or a
known reference and, if stage-2 generalization must be quantified, a separately
designed disjoint-overlap-bin evaluation accounting for pixel-splitting covariance.
No arbitrary scientific pass/fail thresholds are imposed here.

## Scope and tests

This supports the documented EIGER2 CdTe 500K (512 × 1028) and matching energy,
wavelength, thickness, material, orientation and geometry convention. CdTe
absorption is a modeled Beer–Lambert correction, not measured photon-counting QE.
The azimuth wedge reduces azimuth mixing but does not model beam/sample anisotropy.
Response use outside the fit-Q interval is extrapolation. Poisson sigma omits gain,
calibration, monitor and physical-model uncertainty and neighboring-bin covariance.
Treat the method as exploratory until independently validated.

Run the synthetic test suite with `pytest`; it does not require beamline data.
`tox -p` runs the tests together with pre-commit (ruff lint/format) and pyright. The real scan-100605 smoke test (2026-11-10; Python 3.14.6,
pyFAI 2026.5.0) completed in 89.6 s with held-out validation and raw audit: 3608 raw
frames → 41 groups, 334 graph edges, all 2001 bins supported, all JPEGs, no
reported warnings. Absolute held-out overlap changed from 1.054% to 0.247%.
These are a single-run observation, not a speed guarantee or scientific certification.

Raw data and generated products are not versioned. Share them separately subject
to institutional policy; do not commit generated detector data or results.
