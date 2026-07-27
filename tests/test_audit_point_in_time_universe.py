from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_point_in_time_universe.py"
HISTORY_DIR = ROOT / "config" / "vn30_history"


def test_cli_uses_authoritative_history(
    tmp_path: Path,
) -> None:
    market_path = tmp_path / "market_history.csv"
    initial = pd.read_csv(
        HISTORY_DIR / "initial_snapshot.csv"
    )
    changes = pd.read_csv(
        HISTORY_DIR / "membership_changes.csv"
    )
    tickers = sorted(
        set(initial["ticker"]).union(changes["ticker"])
    )
    row_count = len(tickers)

    pd.DataFrame(
        {
            "ticker": tickers,
            "timestamp": ["2016-07-26"] * row_count,
            "open": [10.0] * row_count,
            "high": [11.0] * row_count,
            "low": [9.0] * row_count,
            "close": [10.0] * row_count,
            "volume": [1_000.0] * row_count,
            "value": [10_000.0] * row_count,
        }
    ).to_csv(market_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--history-dir",
            str(HISTORY_DIR),
            "--market-data-path",
            str(market_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        completed.stdout + completed.stderr
    )
    assert "Snapshot rows: 630" in completed.stdout
    assert "Review dates: 21" in completed.stdout
    assert "Membership tickers: 60" in completed.stdout
    assert "Market tickers: 60" in completed.stdout
    assert "Incomplete membership dates: 0" in completed.stdout
    assert "Minimum active constituents: 30" in completed.stdout
    assert "Maximum active constituents: 30" in completed.stdout
    assert (
        "Point-in-time membership coverage passed."
        in completed.stdout
    )


def test_cli_rejects_incomplete_ticker_pool(
    tmp_path: Path,
) -> None:
    market_path = tmp_path / "incomplete_history.csv"
    pd.DataFrame(
        {
            "ticker": ["FPT"],
            "timestamp": ["2016-07-26"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.0],
            "volume": [1_000.0],
            "value": [10_000.0],
        }
    ).to_csv(market_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--history-dir",
            str(HISTORY_DIR),
            "--market-data-path",
            str(market_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    output = completed.stdout + completed.stderr

    assert completed.returncode != 0
    assert (
        "Market data missing historical tickers:"
        in output
    )

def test_help_excludes_stale_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--history-dir" in completed.stdout
    assert "--market-data-path" in completed.stdout
    assert "--membership-path" not in completed.stdout
    assert "data/raw/vnstock" not in completed.stdout
