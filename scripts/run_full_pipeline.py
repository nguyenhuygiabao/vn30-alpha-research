from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str]
    description: str


PIPELINE_STEPS = [
    PipelineStep(
        name="data_quality",
        command=["-m", "src.data_quality"],
        description="Validate raw OHLCV data and refresh the data quality report.",
    ),
    PipelineStep(
        name="features",
        command=["-m", "src.feature_pipeline"],
        description="Build the combined feature dataset.",
    ),
    PipelineStep(
        name="labels",
        command=["-m", "src.labels"],
        description="Build forward relative-return labels.",
    ),
    PipelineStep(
        name="walk_forward_check",
        command=["-m", "src.walk_forward_split"],
        description="Run walk-forward split and purging checks.",
    ),
    PipelineStep(
        name="linear_models",
        command=["-m", "src.linear_models"],
        description="Run linear model experiments.",
    ),
    PipelineStep(
        name="tree_models",
        command=["-m", "src.tree_models"],
        description="Run tree-based model experiments.",
    ),
    PipelineStep(
        name="classification_models",
        command=["-m", "src.classification_models"],
        description="Run classification model experiments.",
    ),
    PipelineStep(
        name="baseline_portfolios",
        command=["-m", "src.baseline_portfolios"],
        description="Run baseline portfolio comparisons.",
    ),
    PipelineStep(
        name="risk_model",
        command=["-m", "src.risk_model"],
        description="Build the past-only rolling risk model.",
    ),
    PipelineStep(
        name="optimizer",
        command=["-m", "src.optimizer"],
        description="Build constrained long-only optimized weights.",
    ),
    PipelineStep(
        name="backtester",
        command=["-m", "src.backtester"],
        description="Run transaction-cost and price-limit-aware backtests.",
    ),
    PipelineStep(
        name="ablation_tests",
        command=["-m", "src.ablation_tests"],
        description="Run feature-ablation tests.",
    ),
    PipelineStep(
        name="horizon_tests",
        command=["-m", "src.horizon_tests"],
        description="Run forecast-horizon tests.",
    ),
    PipelineStep(
        name="dashboard_tables",
        command=[str(ROOT / "scripts" / "build_dashboard_tables.py")],
        description="Build dashboard helper tables from existing processed outputs.",
    ),
    PipelineStep(
        name="model_report",
        command=["-m", "src.model_report"],
        description="Refresh the model report.",
    ),
    PipelineStep(
        name="visualize_results",
        command=["-m", "src.visualize_results"],
        description="Regenerate report figures.",
    ),
    PipelineStep(
        name="interactive_charts",
        command=[str(ROOT / "scripts" / "build_interactive_charts.py")],
        description="Regenerate interactive dashboard charts.",
    ),
    PipelineStep(
        name="html_dashboard",
        command=[str(ROOT / "scripts" / "build_html_report.py")],
        description="Regenerate the HTML dashboard.",
    ),
    PipelineStep(
        name="report_summary",
        command=[str(ROOT / "scripts" / "report_summary.py")],
        description="Print quick report, table, and figure summary.",
    ),
]



EXPENSIVE_ML_STEP_OUTPUTS = {
    "linear_models": [
        ROOT / "data" / "processed" / "linear_model_predictions.parquet",
    ],
    "tree_models": [
        ROOT / "data" / "processed" / "tree_model_predictions.parquet",
        ROOT / "data" / "processed" / "tree_feature_importance.parquet",
    ],
    "classification_models": [
        ROOT / "data" / "processed" / "classification_model_predictions.parquet",
    ],
    "ablation_tests": [
        ROOT / "data" / "processed" / "ablation_tree_predictions.parquet",
        ROOT / "reports" / "tables" / "ablation_results.csv",
    ],
    "horizon_tests": [
        ROOT / "data" / "processed" / "horizon_tree_predictions.parquet",
        ROOT / "reports" / "tables" / "horizon_results.csv",
    ],
}


ML_STEP_DEPENDENCIES = {
    "linear_models": [
        ROOT / "data" / "processed" / "features_combined.parquet",
        ROOT / "data" / "processed" / "labels.parquet",
        ROOT / "src" / "modeling_utils.py",
        ROOT / "src" / "linear_models.py",
        ROOT / "src" / "walk_forward_split.py",
        ROOT / "src" / "metrics.py",
    ],
    "tree_models": [
        ROOT / "data" / "processed" / "features_combined.parquet",
        ROOT / "data" / "processed" / "labels.parquet",
        ROOT / "src" / "modeling_utils.py",
        ROOT / "src" / "tree_models.py",
        ROOT / "src" / "walk_forward_split.py",
        ROOT / "src" / "metrics.py",
    ],
    "classification_models": [
        ROOT / "data" / "processed" / "features_combined.parquet",
        ROOT / "data" / "processed" / "labels.parquet",
        ROOT / "src" / "modeling_utils.py",
        ROOT / "src" / "classification_models.py",
        ROOT / "src" / "walk_forward_split.py",
    ],
    "ablation_tests": [
        ROOT / "data" / "processed" / "features_combined.parquet",
        ROOT / "data" / "processed" / "labels.parquet",
        ROOT / "src" / "modeling_utils.py",
        ROOT / "src" / "ablation_tests.py",
        ROOT / "src" / "tree_models.py",
        ROOT / "src" / "backtester.py",
        ROOT / "src" / "optimizer.py",
    ],
    "horizon_tests": [
        ROOT / "data" / "processed" / "features_combined.parquet",
        ROOT / "data" / "processed" / "labels.parquet",
        ROOT / "src" / "modeling_utils.py",
        ROOT / "src" / "horizon_tests.py",
        ROOT / "src" / "tree_models.py",
        ROOT / "src" / "backtester.py",
        ROOT / "src" / "optimizer.py",
    ],
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VN30 alpha research framework pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pipeline steps without executing them.",
    )
    parser.add_argument(
        "--start-at",
        choices=[step.name for step in PIPELINE_STEPS],
        help="Start from a specific pipeline step.",
    )
    parser.add_argument(
        "--stop-after",
        choices=[step.name for step in PIPELINE_STEPS],
        help="Stop after a specific pipeline step.",
    )
    parser.add_argument(
        "--clean-first",
        action="store_true",
        help="Run the generated-output cleaner before the pipeline.",
    )
    parser.add_argument(
        "--keep-expensive-ml",
        action="store_true",
        help="When used with --clean-first, preserve expensive model prediction/cache parquet files.",
    )
    parser.add_argument(
        "--reuse-existing-ml",
        action="store_true",
        help="Skip expensive ML steps only when their output files exist and are newer than dependency files.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Run only the report summary step.",
    )
    parser.add_argument(
        "--audit-after",
        action="store_true",
        help="Run the project state audit after selected pipeline steps finish.",
    )
    return parser.parse_args()


def selected_steps(start_at: str | None, stop_after: str | None) -> list[PipelineStep]:
    steps = PIPELINE_STEPS

    if start_at is not None:
        start_index = next(index for index, step in enumerate(steps) if step.name == start_at)
        steps = steps[start_index:]

    if stop_after is not None:
        stop_index = next(index for index, step in enumerate(steps) if step.name == stop_after)
        steps = steps[: stop_index + 1]

    return steps



def should_reuse_existing_ml_step(step: PipelineStep) -> bool:
    expected_outputs = EXPENSIVE_ML_STEP_OUTPUTS.get(step.name)

    if expected_outputs is None:
        return False

    dependencies = ML_STEP_DEPENDENCIES.get(step.name, [])

    if not all(path.exists() for path in expected_outputs):
        return False

    if not all(path.exists() for path in dependencies):
        return False

    oldest_output_time = min(path.stat().st_mtime for path in expected_outputs)
    newest_dependency_time = max(path.stat().st_mtime for path in dependencies)

    return oldest_output_time >= newest_dependency_time


def print_reused_ml_step(
    step_number: int,
    total_steps: int,
    step: PipelineStep,
) -> None:
    expected_outputs = EXPENSIVE_ML_STEP_OUTPUTS[step.name]
    dependencies = ML_STEP_DEPENDENCIES.get(step.name, [])

    print()
    print("=" * 80)
    print(f"[{step_number}/{total_steps}] {step.name}")
    print(step.description)
    print("Reusing existing expensive ML outputs.")
    print("Reason: output files exist and are newer than dependency files.")
    print("Expected outputs:")
    for path in expected_outputs:
        print(f"- {path.relative_to(ROOT)}")
    if dependencies:
        print("Dependency files checked:")
        for path in dependencies:
            print(f"- {path.relative_to(ROOT)}")
    print("=" * 80)


def run_step(step_number: int, total_steps: int, step: PipelineStep) -> None:
    print()
    print("=" * 80)
    print(f"[{step_number}/{total_steps}] {step.name}")
    print(step.description)
    print(f"Command: {sys.executable} {' '.join(step.command)}")
    print("=" * 80)

    start_time = time.perf_counter()

    completed = subprocess.run(
        [sys.executable, *step.command],
        cwd=ROOT,
        check=False,
    )

    elapsed = time.perf_counter() - start_time

    if completed.returncode != 0:
        raise RuntimeError(
            f"Pipeline stopped at step {step_number}/{total_steps}: {step.name}. "
            f"Exit code: {completed.returncode}"
        )

    print(f"Finished: {step.name} ({elapsed:.1f} seconds)")


def run_cleaner(dry_run: bool, keep_expensive_ml: bool = False) -> None:
    cleaner_path = ROOT / "scripts" / "clean_generated_outputs.py"

    if not cleaner_path.exists():
        raise FileNotFoundError(f"Missing cleaner script: {cleaner_path}")

    command = [sys.executable, str(cleaner_path)]

    if not dry_run:
        command.append("--confirm")

    if keep_expensive_ml:
        command.append("--keep-expensive-ml")

    print()
    print("=" * 80)
    print("Generated output cleaner")
    print("=" * 80)

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Generated output cleaner failed. Exit code: {completed.returncode}"
        )


def run_audit() -> None:
    audit_path = ROOT / "scripts" / "audit_project_state.py"

    if not audit_path.exists():
        raise FileNotFoundError(f"Missing audit script: {audit_path}")

    print()
    print("=" * 80)
    print("Project state audit")
    print("=" * 80)

    completed = subprocess.run(
        [sys.executable, str(audit_path)],
        cwd=ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Project state audit failed. Exit code: {completed.returncode}"
        )


def main() -> None:
    args = parse_args()
    steps = selected_steps(args.start_at, args.stop_after)
    if args.summary_only and (args.start_at is not None or args.stop_after is not None):
        raise SystemExit("--summary-only cannot be combined with --start-at or --stop-after.")

    if args.summary_only and args.clean_first and not args.dry_run:
        raise SystemExit("--clean-first --summary-only is unsafe without --dry-run.")

    if args.keep_expensive_ml and not args.clean_first:
        raise SystemExit("--keep-expensive-ml requires --clean-first.")

    if args.summary_only:
        steps = [step for step in PIPELINE_STEPS if step.name == "report_summary"]

    print()
    print("VN30 ALPHA RESEARCH PIPELINE")
    print("=" * 80)
    print(f"Project root: {ROOT}")
    print(f"Python executable: {sys.executable}")
    print()
    print("This runner refreshes local outputs using existing raw data.")
    print("It does not download new data.")
    print("It deletes generated parquet outputs only when --clean-first is used.")
    print("Use --keep-expensive-ml with --clean-first to preserve expensive ML caches.")
    print("Use --reuse-existing-ml to skip expensive ML stages only when cached outputs are fresh.")
    print()

    if args.clean_first:
        run_cleaner(
            dry_run=args.dry_run,
            keep_expensive_ml=args.keep_expensive_ml,
        )

    if args.dry_run:
        print("DRY RUN ONLY. No commands will be executed.")
        print()
        for index, step in enumerate(steps, start=1):
            print(f"{index}. {step.name}")
            print(f"   {step.description}")
            if args.reuse_existing_ml and should_reuse_existing_ml_step(step):
                print("   Reuse existing ML outputs: yes")
            else:
                print(f"   Command: {sys.executable} {' '.join(step.command)}")
        print()
        return

    total_steps = len(steps)

    for step_number, step in enumerate(steps, start=1):
        if args.reuse_existing_ml and should_reuse_existing_ml_step(step):
            print_reused_ml_step(step_number, total_steps, step)
            continue

        run_step(step_number, total_steps, step)

    if args.audit_after:
        run_audit()

    print()
    print("=" * 80)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Useful next commands:")
    print("  py .\\scripts\\report_summary.py")
    print("  code .\\reports\\final_results.md")
    print("  code .\\reports\\report_index.md")
    print("  explorer .\\reports\\figures")
    print()


if __name__ == "__main__":
    main()
