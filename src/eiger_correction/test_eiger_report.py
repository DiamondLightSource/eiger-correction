"""The human report is portable, readable, and safe for incomplete records."""

import tempfile
from pathlib import Path
import unittest

from eiger_report import render_markdown, write_markdown_report


class MarkdownReportTests(unittest.TestCase):
    def test_partial_record(self):
        text = render_markdown({"status": "running"})
        self.assertIn("RUNNING", text)
        self.assertIn("not a completed result", text)
        self.assertIn("No completed diagnostic gallery", text)

    def test_summary_metrics_and_portable_image_gallery(self):
        record = {
            "status": "completed_with_warnings", "elapsed_seconds": 89.56,
            "warnings": ["Review support | inspect <mask>"],
            "arguments": {"validation": "holdout", "fit_q_range": [100, 400]},
            "inputs": {"scan": "/source/scan.nxs", "raw_frames": 3608},
            "aggregation": {"positions": 41},
            "integrity_checks": {"status": "passed", "supported_bins": 2001, "unsupported_bins": 0},
            "held_out_validation": [{"fold": 1, "train_positions": [1, 3], "test_positions": [2, 4],
                                     "metrics": [{"label": "detector_only", "absolute_overlap_percent": .247}]}],
            "stages": {"detector_fit": {"status": "passed", "seconds": 16.83}},
            "outputs": {"xy": "pattern_2theta.xy"},
            "diagnostics": {"figures": ["azimuth_geometry.jpg", "plot with spaces.jpg"],
                            "skips": [{"plot": "zoom", "reason": "No supported bins"}]},
        }
        text = render_markdown(record)
        for expected in ("COMPLETED WITH WARNINGS", "3608 → 41", "2001 / 0", "0.247", "16.83",
                         "![Per-position azimuth mask — geometry only](azimuth_geometry.jpg)",
                         "[azimuth_geometry.jpg](azimuth_geometry.jpg)", "plot%20with%20spaces.jpg",
                         "[pattern_2theta.xy](pattern_2theta.xy)", "in-sample", "No supported bins",
                         "Review support \\| inspect &lt;mask&gt;"):
            self.assertIn(expected, text)

    def test_overlap_caption_explains_low_support_and_final_gains(self):
        text = render_markdown({"diagnostics": {"figures": ["overlap_final_abcd.jpg"]}})
        self.assertIn("Low-support spikes", text)
        self.assertIn("very few pixels", text)
        self.assertIn("Both panels use the final applied gains", text)
        self.assertIn("geometric support, not photon counts", text)

    def test_exception_fence_and_atomic_replacement(self):
        record = {"status": "failed", "error": {"type": "ValueError", "message": "bad input",
                                                 "traceback": "Failure with ``` embedded"}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.md"
            write_markdown_report(path, {"status": "running"})
            write_markdown_report(path, record)
            text = path.read_text()
            self.assertIn("FAILED", text)
            self.assertIn("````text", text)
            self.assertIn("ValueError", text)
            self.assertNotIn("**Status: RUNNING**", text)
            self.assertFalse(path.with_suffix(".md.tmp").exists())


if __name__ == "__main__":
    unittest.main()