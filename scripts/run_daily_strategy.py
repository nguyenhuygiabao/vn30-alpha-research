from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class DailyStep:
    name: str
    command: tuple[str, ...]


def build_steps(
    model: str,
    write_paper_plan: bool,
    skip_data_update: bool,
) -> list[DailyStep]:
    steps = []

    if not skip_data_update:
        steps.append(
            DailyStep(
                "daily_data_update",
                (
                    sys.executable,
                    str(ROOT / "scripts" / "run_daily_data_update.py"),
                ),
            )
        )

    steps.extend([
        DailyStep(
            "feature_pipeline",
            (sys.executable, "-m", "src.feature_pipeline"),
        ),
        DailyStep(
            "labels",
            (sys.executable, "-m", "src.labels"),
        ),
        DailyStep(
            "tree_models",
            (sys.executable, "-m", "src.tree_models"),
        ),
        DailyStep(
            "model_health",
            (
                sys.executable,
                str(ROOT / "scripts" / "monitor_model_health.py"),
            ),
        ),
    ])

    planner_command = [
        sys.executable,
        str(ROOT / "scripts" / "preview_daily_paper_orders.py"),
        "--model",
        model,
    ]

    if write_paper_plan:
        planner_command.append("--write")

    steps.append(
        DailyStep(
            "paper_order_plan",
            tuple(planner_command),
        )
    )

    steps.extend([
        DailyStep(
            "dashboard_tables",
            (
                sys.executable,
                str(ROOT / "scripts" / "build_dashboard_tables.py"),
            ),
        ),
        DailyStep(
            "interactive_charts",
            (
                sys.executable,
                str(ROOT / "scripts" / "build_interactive_charts.py"),
            ),
        ),
        DailyStep(
            "html_dashboard",
            (
                sys.executable,
                str(ROOT / "scripts" / "build_html_report.py"),
            ),
        ),
    ])

    return steps


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the guarded VN30 daily strategy workflow."
    )
    parser.add_argument(
        "--model",
        default="rank_ensemble",
        choices=[
            "gradient_boosting",
            "random_forest",
            "rank_ensemble",
        ],
    )
    parser.add_argument(
        "--write-paper-plan",
        action="store_true",
        help="Persist the validated plan to local paper ledgers.",
    )
    parser.add_argument(
        "--skip-data-update",
        action="store_true",
        help="Use only when raw data was already updated and validated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the workflow without running it.",
    )
    return parser.parse_args(argv)


def run_step(
    step: DailyStep,
    number: int,
    total: int,
    runner: Runner,
) -> None:
    print()
    print("=" * 80)
    print(f"[{number}/{total}] {step.name}")
    print("Command:", " ".join(step.command))
    print("=" * 80)

    started = time.perf_counter()
    completed = runner(
        list(step.command),
        cwd=ROOT,
        check=False,
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        raise RuntimeError(
            f"Daily workflow stopped at {step.name}; "
            f"exit code {completed.returncode}"
        )

    print(f"Completed: {step.name} ({elapsed:.1f} seconds)")


def main(
    argv: Sequence[str] | None = None,
    runner: Runner = subprocess.run,
) -> int:
    args = parse_args(argv)
    steps = build_steps(
        model=args.model,
        write_paper_plan=args.write_paper_plan,
        skip_data_update=args.skip_data_update,
    )

    print()
    print("VN30 GUARDED DAILY STRATEGY WORKFLOW")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(
        "Paper-plan mode:",
        "record" if args.write_paper_plan else "preview",
    )
    print("Real-broker connectivity: disabled")

    if args.dry_run:
        print()
        print("DRY RUN")
        for index, step in enumerate(steps, start=1):
            print(f"{index}. {step.name}: {' '.join(step.command)}")
        return 0

    try:
        for index, step in enumerate(steps, start=1):
            run_step(step, index, len(steps), runner)
    except RuntimeError as error:
        print()
        print("DAILY STRATEGY WORKFLOW FAILED")
        print("=" * 80)
        print(error)
        return 1

    print()
    print("DAILY STRATEGY WORKFLOW COMPLETED")
    print("=" * 80)
    print("No real orders were submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
