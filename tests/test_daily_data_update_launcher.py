from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_daily_data_update import (
    PYTHON_314_INSTALL_HELP,
    ROOT,
    build_update_command,
    check_vnstock_quote,
    run_daily_data_update,
)


class FakeRunner:
    def __init__(self, return_codes: list[int]) -> None:
        self.return_codes = iter(return_codes)
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(
        self,
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            next(self.return_codes),
            stdout="",
            stderr="missing dependency",
        )


def test_build_update_command_uses_current_python() -> None:
    command = build_update_command(dry_run=False)

    assert command == [
        sys.executable,
        str(ROOT / "scripts" / "update_daily_data.py"),
    ]
    assert build_update_command(dry_run=True) == [*command, "--dry-run"]


def test_vnstock_preflight_failure_has_python_314_help() -> None:
    runner = FakeRunner([1])

    with pytest.raises(RuntimeError, match="Vnstock Quote preflight failed") as error:
        check_vnstock_quote(runner=runner)

    assert "vnstock==4.0.4" in str(error.value)
    assert PYTHON_314_INSTALL_HELP in str(error.value)
    assert len(runner.calls) == 1


def test_preflight_failure_blocks_the_update() -> None:
    runner = FakeRunner([1])

    with pytest.raises(RuntimeError, match="preflight failed"):
        run_daily_data_update(dry_run=False, runner=runner)

    assert len(runner.calls) == 1


def test_successful_preflight_runs_dry_update() -> None:
    runner = FakeRunner([0, 0])

    run_daily_data_update(dry_run=True, runner=runner)

    assert len(runner.calls) == 2
    assert runner.calls[1][0][-1] == "--dry-run"
    assert runner.calls[1][1] == {"cwd": ROOT, "check": False}


def test_update_failure_is_reported() -> None:
    runner = FakeRunner([0, 7])

    with pytest.raises(RuntimeError, match="Exit code: 7"):
        run_daily_data_update(dry_run=False, runner=runner)


def test_windows_launcher_runs_guarded_python_entrypoint() -> None:
    launcher = Path("run_daily_data_update.cmd").read_text(encoding="utf-8")

    assert "cd /d \"%~dp0\"" in launcher
    assert "py .\\scripts\\run_daily_data_update.py" in launcher
    assert "pause" in launcher
    assert "exit /b %update_exit_code%" in launcher
