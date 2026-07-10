from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

PYTHON_314_INSTALL_HELP = """Python 3.14 setup:
  py -m pip install pydantic openpyxl importlib-metadata tenacity \"vnai>=2.4.8\"
  py -m pip install --no-deps \"vnstock==4.0.4\""""


def check_vnstock_quote(
    runner: CommandRunner = subprocess.run,
) -> None:
    command = [sys.executable, "-c", "from vnstock import Quote"]
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
    message = "Vnstock Quote preflight failed."

    if detail:
        message = f"{message}\n{detail}"

    raise RuntimeError(f"{message}\n\n{PYTHON_314_INSTALL_HELP}")


def build_update_command(dry_run: bool) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / "update_daily_data.py")]

    if dry_run:
        command.append("--dry-run")

    return command


def run_daily_data_update(
    dry_run: bool,
    runner: CommandRunner = subprocess.run,
) -> None:
    check_vnstock_quote(runner=runner)
    command = build_update_command(dry_run=dry_run)
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
    print(f"Python executable: {sys.executable}")
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
