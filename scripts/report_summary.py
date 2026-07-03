from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HORIZON_TABLE = ROOT / "reports" / "tables" / "horizon_results.csv"
ABLATION_TABLE = ROOT / "reports" / "tables" / "ablation_results.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

REPORT_PATHS = [
    ROOT / "README.md",
    ROOT / "reports" / "report_index.md",
    ROOT / "reports" / "final_results.md",
    ROOT / "reports" / "methodology.md",
    ROOT / "reports" / "final_audit.md",
    ROOT / "reports" / "model_report.md",
    ROOT / "reports" / "data_quality_report.md",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def to_float(row: dict[str, str], column: str) -> float:
    return float(row[column])


def print_report_status() -> None:
    print("REPORT FILES")
    print("-" * 60)
    for path in REPORT_PATHS:
        status = "OK" if path.exists() else "MISSING"
        rel_path = path.relative_to(ROOT)
        print(f"{status:8} {rel_path}")
    print()


def print_horizon_summary() -> None:
    rows = read_csv_rows(HORIZON_TABLE)

    best_sharpe = max(rows, key=lambda row: to_float(row, "diagnostic_sharpe"))
    best_rank_ic = max(rows, key=lambda row: to_float(row, "average_rank_ic"))
    best_final_return = max(
        rows,
        key=lambda row: to_float(row, "final_cumulative_after_cost_active_return"),
    )

    print("FORECAST HORIZON SUMMARY")
    print("-" * 60)
    print(
        "Best Sharpe:"
        f" {best_sharpe['forecast_horizon_days']}d"
        f" ({to_float(best_sharpe, 'diagnostic_sharpe'):.6f})"
    )
    print(
        "Best Rank IC:"
        f" {best_rank_ic['forecast_horizon_days']}d"
        f" ({to_float(best_rank_ic, 'average_rank_ic'):.6f})"
    )
    print(
        "Best final after-cost active return:"
        f" {best_final_return['forecast_horizon_days']}d"
        f" ({to_float(best_final_return, 'final_cumulative_after_cost_active_return'):.6f})"
    )
    print()


def print_ablation_summary() -> None:
    rows = read_csv_rows(ABLATION_TABLE)

    best_sharpe = max(rows, key=lambda row: to_float(row, "diagnostic_sharpe"))
    best_rank_ic = max(rows, key=lambda row: to_float(row, "average_rank_ic"))
    best_final_return = max(
        rows,
        key=lambda row: to_float(row, "final_cumulative_after_cost_active_return"),
    )

    print("FEATURE ABLATION SUMMARY")
    print("-" * 60)
    print(
        "Best Sharpe:"
        f" {best_sharpe['ablation_name']}"
        f" ({to_float(best_sharpe, 'diagnostic_sharpe'):.6f})"
    )
    print(
        "Best Rank IC:"
        f" {best_rank_ic['ablation_name']}"
        f" ({to_float(best_rank_ic, 'average_rank_ic'):.6f})"
    )
    print(
        "Best final after-cost active return:"
        f" {best_final_return['ablation_name']}"
        f" ({to_float(best_final_return, 'final_cumulative_after_cost_active_return'):.6f})"
    )
    print()


def print_figure_summary() -> None:
    figures = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    non_empty = [path for path in figures if path.stat().st_size > 0]

    print("FIGURE SUMMARY")
    print("-" * 60)
    print(f"PNG figures found: {len(figures)}")
    print(f"Non-empty PNG figures: {len(non_empty)}")

    for path in figures:
        size_kb = path.stat().st_size / 1024
        print(f"- {path.name} ({size_kb:.1f} KB)")
    print()


def main() -> None:
    print()
    print("VN30 LONG-ONLY ML FRAMEWORK SUMMARY")
    print("=" * 60)
    print()

    print_report_status()
    print_horizon_summary()
    print_ablation_summary()
    print_figure_summary()

    print("KEY PATHS")
    print("-" * 60)
    print("README: reports are linked from README.md")
    print("Report index: reports/report_index.md")
    print("Main results: reports/final_results.md")
    print("Methodology: reports/methodology.md")
    print("Final audit: reports/final_audit.md")
    print()


if __name__ == "__main__":
    main()
