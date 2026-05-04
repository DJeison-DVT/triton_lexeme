"""
conftest.py — Build the scanner binary before running tests.
"""
import subprocess
import os
import pytest

SCANNER_DIR = os.path.dirname(os.path.abspath(__file__))
SCANNER_BIN = os.path.join(SCANNER_DIR, "triton_scanner")


def pytest_configure(config):
    """Ensure the scanner binary is built before any test runs."""
    if not os.path.isfile(SCANNER_BIN):
        print("Building scanner...")
        result = subprocess.run(
            ["make", "-C", SCANNER_DIR],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            pytest.exit(f"Failed to build scanner:\n{result.stderr}")
