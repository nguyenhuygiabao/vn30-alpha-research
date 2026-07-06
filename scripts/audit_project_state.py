from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

KEY_FILES = [
    ROOT / "README.md",
    ROOT / "reports" / "report_index.md",
    ROOT / "reports" / "final_results.md",
    ROOT / "reports" / "methodology.md",
    ROOT / "reports" / "final_audit.md",
    ROOT / "reports" / "model_report.md",
    ROOT / "reports" / "data_quality_report.md",
    ROOT / "reports" / "dashboard.html",
    ROOT / "reports" / "interactive" / "interactive_cumulative_return.html",
    ROOT / "reports" / "interactive" / "interactive_active_drawdown.html",
    ROOT / "reports" / "interactive" / "interactive_portfolio_turnover.html",
    ROOT / "reports" / "interactive" / "interactive_rolling_diagnostic_sharpe.html",
    ROOT / "reports" / "tables" / "ablation_results.csv",
    ROOT / "reports" / "tables" / "horizon_results.csv",
    ROOT / "reports" / "tables" / "benchmark_results.csv",
    ROOT / "reports" / "tables" / "concentration_summary.csv",
    ROOT / "reports" / "tables" / "issuer_group_exposure_latest.csv",
    ROOT / "reports" / "tables" / "latest_rank_diagnostic.csv",
    ROOT / "reports" / "tables" / "horizon_sample_disclosure.csv",
    ROOT / "scripts" / "run_full_pipeline.py",
    ROOT / "scripts" / "clean_generated_outputs.py",
    ROOT / "scripts" / "report_summary.py",
    ROOT / "scripts" / "audit_project_state.py",
    ROOT / "scripts" / "build_html_report.py",
    ROOT / "scripts" / "build_interactive_charts.py",
    ROOT / "scripts" / "build_dashboard_tables.py",
    ROOT / "scripts" / "darken_static_figures.py",
]

GENERATED_PATH_PREFIXES = [
    "data/processed/",
    "data/raw/vnstock/",
]


def run_git_status() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())

    return completed.stdout.splitlines()


def normalize_status_path(status_line: str) -> str:
    return status_line[3:].replace("\\", "/")


def main() -> None:
    print()
    print("VN30 PROJECT STATE AUDIT")
    print("=" * 80)
    print(f"Project root: {ROOT}")
    print()

    failed = False

    print("KEY FILES")
    print("-" * 80)
    for path in KEY_FILES:
        exists = path.exists()
        non_empty = path.is_file() and path.stat().st_size > 0

        if exists and non_empty:
            print(f"OK       {path.relative_to(ROOT)}")
        else:
            failed = True
            print(f"MISSING  {path.relative_to(ROOT)}")

    print()
    print("FIGURES")
    print("-" * 80)
    figure_dir = ROOT / "reports" / "figures"
    png_files = sorted(figure_dir.glob("*.png")) if figure_dir.exists() else []
    non_empty_png_files = [path for path in png_files if path.stat().st_size > 0]

    print(f"PNG figures found: {len(png_files)}")
    print(f"Non-empty PNG figures: {len(non_empty_png_files)}")

    if not png_files or len(png_files) != len(non_empty_png_files):
        failed = True
        print("WARNING  Missing or empty PNG figures detected.")
    else:
        print("OK       All PNG figures are non-empty.")

    print()
    print("GIT STATUS")
    print("-" * 80)
    status_lines = run_git_status()

    if not status_lines:
        print("OK       Working tree is clean.")
    else:
        print("Working tree has changes:")
        for line in status_lines:
            print(f"- {line}")

    leaked_generated_files = []
    for line in status_lines:
        path = normalize_status_path(line)
        if any(path.startswith(prefix) for prefix in GENERATED_PATH_PREFIXES):
            leaked_generated_files.append(path)

    print()
    print("GENERATED DATA LEAK CHECK")
    print("-" * 80)
    if leaked_generated_files:
        failed = True
        print("FAILED   Generated data files appear in Git status:")
        for path in leaked_generated_files:
            print(f"- {path}")
    else:
        print("OK       No generated raw/processed data files appear in Git status.")

    print()
    print("=" * 80)
    if failed:
        print("AUDIT FAILED")
        raise SystemExit(1)

    print("AUDIT PASSED")
    print()


if __name__ == "__main__":
    main()
