from __future__ import annotations

from scripts.execute_pending_paper_orders import parse_args


def test_execution_cli_defaults_to_preview() -> None:
    args = parse_args([
        "--execution-date",
        "2026-07-22",
    ])

    assert args.execution_date == "2026-07-22"
    assert args.write is False


def test_execution_cli_requires_explicit_write() -> None:
    args = parse_args([
        "--execution-date",
        "2026-07-22",
        "--write",
    ])

    assert args.write is True
