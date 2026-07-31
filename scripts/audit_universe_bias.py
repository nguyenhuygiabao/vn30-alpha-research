from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_pipeline import (
    MEMBERSHIP_HISTORY_DIR,
    load_verified_universe_reconstruction,
)
from src.research_market_data import (
    DEFAULT_ADJUSTED_MARKET_PATH,
    DEFAULT_RAW_MARKET_PATH,
    load_research_market_history,
)
from src.universe_bias import (
    assert_complete_membership_interval_data,
    build_membership_interval_audit,
    detect_current_only_history_bias,
)
from src.universe_history import (
    snapshots_to_membership_history,
    validate_constituent_snapshot_coverage,
)

DEFAULT_REPORT_PATH = "reports/tables/universe_bias_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit historical VN30 universe coverage and check the market "
            "data for a current-members-only survivorship signature."
        ),
    )
    parser.add_argument(
        "--raw-market-path",
        default=DEFAULT_RAW_MARKET_PATH,
        help="FiinQuant canonical raw OHLC history.",
    )
    parser.add_argument(
        "--adjusted-market-path",
        default=DEFAULT_ADJUSTED_MARKET_PATH,
        help="FiinQuant canonical adjusted history.",
    )
    parser.add_argument(
        "--membership-dir",
        default=MEMBERSHIP_HISTORY_DIR,
        help="Directory holding verified VN30 membership history.",
    )
    parser.add_argument("--output", default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # This audit deliberately loads the unfiltered vendor history. Applying
    # point-in-time membership first would remove exactly the former-member
    # rows the bias check is looking for.
    market = load_research_market_history(
        args.raw_market_path,
        args.adjusted_market_path,
        point_in_time=False,
    )
    reconstruction = load_verified_universe_reconstruction(
        args.membership_dir
    )

    verified_snapshots = validate_constituent_snapshot_coverage(
        snapshots=reconstruction.snapshots,
        start_date=market["date"].min(),
        end_date=market["date"].max(),
        expected_size=30,
        maximum_gap_days=220,
    )
    membership = snapshots_to_membership_history(
        verified_snapshots,
        expected_size=30,
    )

    audit = build_membership_interval_audit(
        market_data=market,
        membership=membership,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)

    missing_intervals = int(audit["missing_interval"].sum())
    boundary_reviews = int(
        audit["coverage_status"].eq("boundary_review").sum()
    )
    current_only_bias = detect_current_only_history_bias(
        market_data=market,
        membership=membership,
    )

    print("HISTORICAL VN30 UNIVERSE BIAS AUDIT")
    print("=" * 80)
    print("Raw-market source:", args.raw_market_path)
    print("Adjusted-market source:", args.adjusted_market_path)
    print("Membership source:", args.membership_dir)
    print("Market rows:", len(market))
    print("Market tickers:", market["ticker"].nunique())
    print("Snapshot dates:", verified_snapshots["effective_date"].nunique())
    print("Historical tickers:", membership["ticker"].nunique())
    print("Membership intervals:", len(audit))
    print("Missing intervals:", missing_intervals)
    print("Boundary reviews:", boundary_reviews)
    print("Current-only vendor signature:", current_only_bias)
    print("Audit report:", output_path)

    assert_complete_membership_interval_data(audit)

    if current_only_bias:
        raise RuntimeError(
            "Market data contain current members but no former VN30 members"
        )

    print("Bias-control gate: PASS")


if __name__ == "__main__":
    main()
