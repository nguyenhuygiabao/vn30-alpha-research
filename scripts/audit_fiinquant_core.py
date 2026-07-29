from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_pipeline import validate_combined_features
from src.features_foreign_flow import (
    load_canonical_foreign_flow_csv,
)
from src.market_data_schema import (
    load_canonical_market_history,
)

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_CORE_FILES = (
    "config/exchange_history.csv",
    "scripts/audit_point_in_time_universe.py",
    "scripts/download_fiinquant_history.py",
    "scripts/import_fiinquant_archived_history.py",
    "src/corporate_actions.py",
    "src/exchange_history.py",
    "src/exchange_rules.py",
    "src/feature_pipeline.py",
    "src/features_foreign_flow.py",
    "src/features_herding.py",
    "src/features_liquidity.py",
    "src/fiinquant_adapter.py",
    "src/market_data_schema.py",
    "src/vn30_universe.py",
)

RAW_MARKET_PATH = (
    ROOT
    / "data"
    / "raw"
    / "fiinquant"
    / "canonical"
    / "raw_market_history.csv"
)
ADJUSTED_MARKET_PATH = (
    ROOT
    / "data"
    / "raw"
    / "fiinquant"
    / "canonical"
    / "adjusted_market_history.csv"
)
FOREIGN_FLOW_PATH = (
    ROOT
    / "data"
    / "raw"
    / "fiinquant"
    / "canonical"
    / "foreign_flow_history.csv"
)
FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "features_combined.parquet"
)


def require_core_files() -> None:
    missing = [
        path
        for path in REQUIRED_CORE_FILES
        if not (ROOT / path).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"Missing FiinQuant core files: {missing}"
        )


def reject_eager_sdk_import() -> None:
    downloader = (
        ROOT
        / "scripts"
        / "download_fiinquant_history.py"
    )
    tree = ast.parse(
        downloader.read_text(encoding="utf-8")
    )

    for node in tree.body:
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}

            if "FiinQuantX" in names:
                raise RuntimeError(
                    "The production downloader imports "
                    "FiinQuantX at module load time"
                )

        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "FiinQuantX"
        ):
            raise RuntimeError(
                "The production downloader imports "
                "FiinQuantX at module load time"
            )


def audit_canonical_data(
    *,
    require_data: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    paths = (
        RAW_MARKET_PATH,
        ADJUSTED_MARKET_PATH,
        FOREIGN_FLOW_PATH,
    )
    present = [path.is_file() for path in paths]

    if not any(present):
        if require_data:
            raise RuntimeError(
                "Canonical FiinQuant data are required "
                "for this audit"
            )

        print("Canonical data: skipped")
        return None, None

    if not all(present):
        missing = [
            str(path.relative_to(ROOT))
            for path, exists in zip(
                paths,
                present,
                strict=True,
            )
            if not exists
        ]
        raise RuntimeError(
            "Canonical FiinQuant data are incomplete: "
            f"{missing}"
        )

    market = load_canonical_market_history(
        RAW_MARKET_PATH,
        ADJUSTED_MARKET_PATH,
    )
    foreign_flow = load_canonical_foreign_flow_csv(
        FOREIGN_FLOW_PATH
    )

    market_keys = pd.MultiIndex.from_frame(
        market[["date", "ticker"]]
    )
    flow_keys = pd.MultiIndex.from_frame(
        foreign_flow[["date", "ticker"]]
    )

    if not market_keys.equals(flow_keys):
        raise RuntimeError(
            "Canonical market and foreign-flow keys differ"
        )

    print(f"Canonical market rows: {len(market)}")
    print(
        "Canonical market tickers: "
        f"{market['ticker'].nunique()}"
    )
    print(
        "Raw-price fallback rows: "
        f"{market.attrs['raw_price_fallback_rows']}"
    )
    print(
        "Canonical foreign-flow rows: "
        f"{len(foreign_flow)}"
    )
    return market, foreign_flow


def audit_feature_output(
    market: pd.DataFrame | None,
    *,
    require_data: bool,
) -> None:
    if not FEATURE_PATH.is_file():
        if require_data:
            raise RuntimeError(
                "Combined feature output is required "
                "for this audit"
            )

        print("Combined features: skipped")
        return

    features = pd.read_parquet(FEATURE_PATH)
    audit = validate_combined_features(features)

    if (
        market is not None
        and features["date"].max() > market["date"].max()
    ):
        raise RuntimeError(
            "Feature output extends beyond market history"
        )

    print(f"Combined feature rows: {len(features)}")
    print(
        "Combined feature columns: "
        f"{len(features.columns)}"
    )
    print(
        "Duplicate feature keys: "
        f"{audit['duplicate_keys']}"
    )
    print(
        "Infinite feature columns: "
        f"{audit['infinite_columns']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the isolated FiinQuant production core."
        )
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help=(
            "Require canonical data and rebuilt features "
            "instead of allowing a code-only audit."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("FIINQUANT CORE AUDIT")
    print("=" * 72)
    require_core_files()
    reject_eager_sdk_import()
    market, _ = audit_canonical_data(
        require_data=args.require_data
    )
    audit_feature_output(
        market,
        require_data=args.require_data,
    )
    print("=" * 72)
    print("FIINQUANT CORE AUDIT PASSED")


if __name__ == "__main__":
    main()
