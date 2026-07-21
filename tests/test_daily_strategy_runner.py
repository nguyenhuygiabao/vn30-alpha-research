from __future__ import annotations

import subprocess

from scripts.run_daily_strategy import build_steps, main


def test_default_daily_workflow_ends_with_preview() -> None:
    steps = build_steps(
        model="rank_ensemble",
        write_paper_plan=False,
        skip_data_update=False,
    )

    assert steps[0].name == "daily_data_update"
    assert steps[-4].name == "paper_order_plan"
    assert "--write" not in steps[-4].command
    assert [step.name for step in steps[-3:]] == [
        "dashboard_tables",
        "interactive_charts",
        "html_dashboard",
    ]


def test_recording_requires_explicit_flag() -> None:
    steps = build_steps(
        model="rank_ensemble",
        write_paper_plan=True,
        skip_data_update=True,
    )

    assert steps[0].name == "feature_pipeline"
    assert steps[-4].command[-1] == "--write"


def test_daily_workflow_stops_after_failed_step() -> None:
    calls = []

    def runner(command, cwd, check):
        del cwd, check
        calls.append(command)

        return subprocess.CompletedProcess(
            command,
            1 if "src.labels" in command else 0,
        )

    result = main(
        ["--skip-data-update"],
        runner=runner,
    )

    assert result == 1
    assert len(calls) == 2
