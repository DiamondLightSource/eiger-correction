"""Interface for ``python -m eiger_correction``."""

from __future__ import annotations

import logging
import traceback
import warnings
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

from eiger_correction.extract_eiger_corrections import RunRecord, extract, parser

from . import __version__

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    cli = parser()
    cli.add_argument("-v", "--version", action="version", version=__version__)
    args = cli.parse_args(argv)
    directory = (
        args.output_dir
        or Path("outputs") / f"{args.scan.stem}_{datetime.now(UTC):%Y%m%dT%H%M%S_%fZ}"
    ).resolve()
    # Exclusive directory creation prevents interleaved jobs and output clobbering.
    directory.mkdir(parents=True, exist_ok=False)
    record = RunRecord(directory, args)
    with (directory / "run.log").open("x", encoding="utf-8", buffering=1) as log:
        handler = logging.StreamHandler(log)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root = logging.getLogger()
        root.addHandler(handler)
        logging.captureWarnings(True)
        try:
            with (
                redirect_stdout(log),
                redirect_stderr(log),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                try:
                    extract(args, directory, record)
                    record.data["status"] = "completed"
                except BaseException as error:
                    record.data.update(
                        status="interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed",
                        error={
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    traceback.print_exc(file=log)
                finally:
                    record.data["warnings"].extend(
                        f"{item.category.__name__}: {item.message}" for item in caught
                    )
                    if record.data["status"] == "completed" and record.data["warnings"]:
                        record.data["status"] = "completed_with_warnings"
                    record.data["finished_utc"] = datetime.now(UTC).isoformat()
                    record.save()
        finally:
            logging.captureWarnings(False)
            root.removeHandler(handler)
    return 0 if record.data["status"].startswith("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
