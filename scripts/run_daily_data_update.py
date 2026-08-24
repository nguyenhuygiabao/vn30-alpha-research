from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

# FiinQuantX is a vendor wheel and is not installed in the interpreter that
# runs the research pipeline. It lives in a dedicated environment, so the
# download step is launched with that interpreter when it is present.
FIINQUANT_INSTALL_HELP = """FiinQuant setup:
  py -m venv .venv-fiinquant
  .\\.venv-fiinquant\\Scripts\\python.exe -m pip install FiinQuantX pandas pyarrow PyYAML

Credentials are read from the FIINQUANT_USERNAME and FIINQUANT_PASSWORD
environment variables, and prompted for interactively when either is unset."""


def resolve_python_executable(root: Path = ROOT) -> str:
    """Return the interpreter that can import FiinQuantX."""

    candidates = (
        root / ".venv-fiinquant" / "Scripts" / "python.exe",
        root / ".venv-fiinquant" / "bin" / "python",
    )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return sys.executable


def check_fiinquant_client(
    runner: CommandRunner = subprocess.run,
    python_executable: str | None = None,
) -> None:
    executable = python_executable or resolve_python_executable()
    command = [executable, "-c", "from FiinQuantX import FiinSession"]
    completed = runner(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode == 0:
        return

    detail = (completed.stderr or completed.stdout or "").strip()
    message = "FiinQuant client preflight failed."

    if detail:
        message = f"{message}\n{detail}"

    raise RuntimeError(f"{message}\n\n{FIINQUANT_INSTALL_HELP}")


def build_update_command(
    dry_run: bool,
    python_executable: str | None = None,
) -> list[str]:
    executable = python_executable or resolve_python_executable()
    command = [executable, str(ROOT / "scripts" / "update_daily_data.py")]

    if dry_run:
        command.append("--dry-run")

    return command


def run_daily_data_update(
    dry_run: bool,
    runner: CommandRunner = subprocess.run,
) -> None:
    executable = resolve_python_executable()
    check_fiinquant_client(runner=runner, python_executable=executable)
    command = build_update_command(
        dry_run=dry_run,
        python_executable=executable,
    )
    completed = runner(command, cwd=ROOT, check=False)

    if completed.returncode != 0:
        raise RuntimeError(
            "Daily data update command failed. Review the output above. "
            f"Exit code: {completed.returncode}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the guarded VN30 daily data update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and validate without replacing the local raw file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    print()
    print("VN30 ONE-CLICK DAILY DATA UPDATE")
    print("=" * 80)
    print(f"Project root: {ROOT}")
    print(f"Launcher interpreter: {sys.executable}")
    print(f"FiinQuant interpreter: {resolve_python_executable()}")
    print(f"Mode: {'dry run' if args.dry_run else 'validated file update'}")
    print("No paper orders or real orders will be placed.")
    print()

    try:
        run_daily_data_update(dry_run=args.dry_run)
    except (OSError, RuntimeError) as error:
        print()
        print("DAILY DATA UPDATE FAILED")
        print("=" * 80)
        print(error)
        print()
        return 1

    print("ONE-CLICK DAILY DATA UPDATE COMPLETED SUCCESSFULLY")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
