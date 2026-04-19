"""Pytest configuration — ensures tests run against the latest grammar."""

import subprocess
import warnings

import pytest


def _check_branch_freshness():
    """Warn if local branch is behind origin/main."""
    try:
        subprocess.run(
            ["git", "fetch", "origin", "main", "--quiet"],
            capture_output=True,
            timeout=10,
        )
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        behind = int(result.stdout.strip())
        if behind > 0:
            return behind
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0


def pytest_configure(config):
    behind = _check_branch_freshness()
    if behind:
        warnings.warn(
            f"\n\n*** Your branch is {behind} commit(s) behind origin/main. ***\n"
            f"*** Run 'git pull origin main' before testing the grammar. ***\n",
            stacklevel=1,
        )
