from pathlib import Path
import pandas as pd
from .data_loader import load_ohlcv_csv

SOURCE_PATH = "data/raw/vnstock/vn30_ohlcv.csv"
REPORT_PATH = "reports/data_quality_report.md"
PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close"
]

JUMP_THRESHOLD = 0.1  # 10% price jump threshold
STALE_RUN_LENGTH = 3
MAX_DETAILED_ISSUES = 200

#The function below receives a pandas dataframe as an input as well as the custom jump and stale price thresholds. This function will return another dataframe that contains the problems it found.
def collect_data_issues(data: pd.DataFrame, jump_threshold: float = JUMP_THRESHOLD, stale_run_length: int = STALE_RUN_LENGTH,
) -> pd.DataFrame:
    """Find suspicious or invalid observations in the OHLCV data."""

    working = (
        data
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
        .copy()
    )
    issue_frames: list[pd.DataFrame] = []
    def add_issue(
            mask: pd.Series, issue_name: str, detail_columns: list[str],
    ) -> None:
        """Store rows that match a data-quality condition"""

        if not mask.any():
            return
        flagged = working.loc[mask, ["date", "ticker"] + detail_columns].copy()

        flagged["issue"] = issue_name

        flagged["details"] = (flagged[detail_columns]
                              .astype(str)
            .agg(", ".join, axis=1)
        )

        issue_frames.append(
            flagged[["date",
                     "ticker",
                     "issue",
                     "details"]]
        )
    # Check whether any column in a row contains a missing value.
    # For every row, check whether any column contains a missing value, working.isna() creates a table of True and False values, axis = 1 check horizontally across the columns
    missing_mask = working.isna().any(axis=1)
    if missing_mask.any():
        flagged = working.loc[
            missing_mask, ["date", "ticker"], ].copy()

        flagged["issue"] = "missing_values"

        flagged["details"] = working.loc[missing_mask
        ].apply(
            lambda row: (
                "missing:"
                + ",".join(row.index[row.isna()])
            ),
        axis = 1,
        )
        issue_frames.append(
            flagged[["date",
                     "ticker",
                     "issue",
                     "details"]]
        )
    #Check for more than one observation for the same ticker and date.
    #Check whether the same ticker appears more than once on the same date.
    duplicate_mask = working.duplicated(
        subset=["date", "ticker"],
        keep=False,
    )

    add_issue(
        duplicate_mask,
        "duplicate_ticker_date",
        ["open",
         "high",
         "low",
         "close"],
    )

    #Check for non positive volume
    add_issue(
        working["volume"] <= 0,
        "non_positive_volume",
        ["volume"],
    )

    #Check for non positive traded value
    add_issue(
        working["value_traded"] <= 0,
        "non_positive_value_traded",
        ["value_traded"],
    )

    #Check for non positive prices
    add_issue(
        (working[PRICE_COLUMNS] <= 0).any(axis=1),
        "non_positive_price",
        PRICE_COLUMNS,
    )

    #Making sure that high price >= low price
    add_issue(
        working["high"] < working["low"],
        "high_below_low",
        ["high", "low"],
    )

    # Flag opening prices outside the daily high-low range.
    add_issue(
        (working["open"] < working["low"]) | (working["open"] > working["high"]),
        "open_outside_daily_range",
        ["open", "low", "high"],
    )

    # Flag closing prices outside the daily high-low range.
    add_issue(
        (working["close"] < working["low"]) | (working["close"] > working["high"]),
        "close_outside_daily_range",
        ["close", "low", "high"],
    )

    #Check for suspicious price jumps. This is done by calculating the percentage change in adjusted close price for each ticker and flagging any changes that exceed the jump threshold.
    working["adjusted_return_1d"] = (
        working
        .groupby("ticker")["adjusted_close"] #Group by ticker separates the rows by stock
        .pct_change(fill_method = None) #adjusted_close.pct_change() calculates the percentage change in adjusted close price for each ticker.
    )
    add_issue(
        working["adjusted_return_1d"].abs() > jump_threshold,
        "suspicious_adjusted_close_jump",
        ["adjusted_close", "adjusted_return_1d"],
    )
    #For each ticker, compare every adjusted close with the previous adjusted close, and assign an ID to each consecutive run of equal prices.
    working["price_run_id"] = (
        working
        .groupby("ticker")["adjusted_close"]
        .transform(
            lambda series: (
                series
                .ne(series.shift()) #ne() compares each value in the series to the previous value (shifted by one position) and returns True if they are not equal, indicating a change in price.
                .cumsum() #cumsum() calculates the cumulative sum of the boolean values, effectively assigning a unique run ID to each sequence of identical prices.
            )
        )
    )
    #Group rows by ticker and price-run ID, count the number of rows in each run, and put that count back onto every row.
    working["stale_run_length"] = (
        working
        .groupby(["ticker", "price_run_id"])["adjusted_close"]
        .transform("size")
    )
    #Flag any rows where the run length of identical adjusted close prices is greater than or equal to the stale run length threshold, as these may indicate stale prices.
    add_issue(
        working["stale_run_length"] >= stale_run_length,
        "stale_adjusted_close",
        [
            "adjusted_close",
            "stale_run_length"
        ]
    )
    #If no problems were found, return an empty dataframe with the expected column structure.
    if not issue_frames:
        return pd.DataFrame(
            columns =[ "date",
                      "ticker",
                      "issue",
                      "details"
                      ]
        )
    #Combine all the issue DataFrames into a single table.
    issues = pd.concat(issue_frames, ignore_index=True)

    #Put the issues into a predictable order.
    issues = issues.sort_values(["date",
                                 "ticker",
                                 "issue"]).reset_index(drop=True
    )

    return issues


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table string."""

    if data.empty:
        return "_No rows to display_"

    display = data.copy()

    for column in display.columns:
        if pd.api.types.is_datetime64_any_dtype(
            display[column]
        ):
            display[column] = display[column].dt.strftime("%Y-%m-%d")

    headers = [str(column) for column in display.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in display.itertuples(
        index = False,
        name = None,
    ):
        values = []

        for value in row:
            if pd.isna(value):
                text = ""
            else:
                text = str(value)

            text = (
                text
                .replace("|", "\\|")
                .replace("\n", "")
            )
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)





def write_data_quality_report(
    data: pd.DataFrame,
    issues: pd.DataFrame,
    source_path: str,
    report_path: str,
) -> None:
    """Create a Markdown report from data-quality results."""

    output_path = Path(report_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_at = pd.Timestamp.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if data.empty:
        earliest_date = "N/A"
        latest_date = "N/A"
    else:
        earliest_date = data["date"].min().strftime(
            "%Y-%m-%d"
        )

        latest_date = data["date"].max().strftime(
            "%Y-%m-%d"
        )
    if issues.empty:
        status = "PASS"


        issue_counts = pd.DataFrame(
            columns = ["issue","flag_count"]
        )

        ticker_counts = pd.DataFrame(
            columns = ["ticker","flag_count"]
        )

        date_counts = pd.DataFrame(
            columns = ["date", "flag_count"]
        )
    else:
        status = "REVIEW REQUIRED"

        issue_counts = (
            issues
            .groupby("issue")
            .size()
            .reset_index(name = "flag_count")
            .sort_values(
                "flag_count",
                ascending= False,
            )
        )

        ticker_counts = (
            issues
            .groupby("ticker")
            .size()
            .reset_index(name="flag_count")
            .sort_values(
                "flag_count",
                ascending=False,
            )
        )

        date_counts = (
            issues
            .groupby("date")
            .size()
            .reset_index(name = "flag_count")
            .sort_values("date")
        )

    detailed_issues = issues.head(MAX_DETAILED_ISSUES).copy()

    if len(issues) > MAX_DETAILED_ISSUES:
        detailed_issue_note = (
            f"Showing first {MAX_DETAILED_ISSUES} issue rows "
            f"out of {len(issues)} total flags."
        )
    else:
        detailed_issue_note = (
            f"Showing all {len(issues)} issue rows."
        )
        detailed_issues = issues.head(MAX_DETAILED_ISSUES).copy()

    if len(issues) > MAX_DETAILED_ISSUES:
        detailed_issue_note = (
            f"Showing first {MAX_DETAILED_ISSUES} issue rows "
            f"out of {len(issues)} total flags."
        )
    else:
        detailed_issue_note = (
            f"Showing all {len(issues)} issue rows."
        )

    report_lines = [
        "# VN30 Data Quality Report",
        "",
        "## Overall Status",
        "",
        f"**{status}**",
        "",
        "A flag identifies a row requiring review. "
        "A single source row may receive multiple flags.",
        "",
        "## Dataset Summary",
        "",
        f"- Source file: `{source_path}`",
        f"- Report generated: {generated_at}",
        f"- Number of rows: {len(data)}",
        f"- Number of tickers: {data['ticker'].nunique()}",
        f"- Earliest date: {earliest_date}",
        f"- Latest date: {latest_date}",
        f"- Total issue flags: {len(issues)}",
        "",
        "## Issue Counts by Type",
        "",
        dataframe_to_markdown(issue_counts),
        "",
        "## Flag Counts by Ticker",
        "",
        dataframe_to_markdown(ticker_counts),
        "",
        "## Flag Counts by Date",
        "",
        dataframe_to_markdown(date_counts),
        "",
        "## Detailed Problem Rows",
        "",
        detailed_issue_note,
        "",
        dataframe_to_markdown(detailed_issues),
        "",
        "## Treatment Policy", \
        "",
        "- Missing date or ticker: drop the row.",
        "- Missing price: exclude until the source is verified.",
        "- Exact duplicate: retain one copy.",
        "- Conflicting duplicate: investigate manually.",
        "- Zero volume: retain but mark as non-tradable.",
        "- Negative volume: correct from source or drop.",
        "- Non-positive price: correct from source or drop.",
        "- Invalid OHLC relationship: correct or drop.",
        "- Suspicious jump: investigate corporate actions.",
        "- Stale price: investigate liquidity, suspension, "
        "or vendor errors.",
        "",
        "## Current Limitations",
        "",
        "- The current data is vendor-sourced daily OHLCV data from vnstock and should still be checked against official exchange records when possible.",
        "- Corporate actions are not independently verified.",
        "- Historical VN30 membership is not yet included.",
        "- Survivorship bias remains unresolved.",
        "- Official exchange-provided ceiling and floor prices are not included; estimated price-limit features are generated separately and should be checked against official data when available.",
        "",
    ]

    output_path.write_text(
        "\n".join(report_lines),
        encoding = "utf-8",
    )

def main() -> None:
    """Run the data-quality pipeline"""

    data = load_ohlcv_csv(SOURCE_PATH)

    issues = collect_data_issues(data)

    write_data_quality_report(
        data = data,
        issues = issues,
        source_path= SOURCE_PATH,
        report_path = REPORT_PATH,
    )

    print("Data-quality checks completed.")
    print(f"Rows checked: {len(data)}")
    print(f"Tickers checked: {data['ticker'].nunique()}")
    print(f"Issues flags found: {len(issues)}")
    print(f"Reports created: {REPORT_PATH}")

if __name__ == "__main__":
    main()
