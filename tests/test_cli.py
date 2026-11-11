import subprocess
import sys

from eiger_correction import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "eiger_correction", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
