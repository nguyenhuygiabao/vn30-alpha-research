"""Daily VN30 OHLCV for the paper-trading update, sourced from FiinQuant.

The research pipeline runs entirely on FiinQuant. This provider puts the live
daily update on the same vendor so the paper-trading file cannot drift away
from the data the models were fitted on.

The file this feeds follows the same raw/adjusted contract as the research
pipeline, so each window needs two vendor requests:

* ``open``, ``high``, ``low``, ``close`` come from the raw series and drive
  execution and price limits.
* ``adjusted_close`` comes from the adjusted series and drives returns.

Prices are stored in thousands of VND while FiinQuant reports whole VND, so
every price column is divided by ``FIINQUANT_PRICE_DIVISOR``.

The vendor endpoint is batch-oriented, so the first call for a window fetches
the whole universe and subsequent per-ticker calls slice that frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from src.fiinquant_adapter import FIINQUANT_PRICE_DIVISOR


RAW_FIELDS = ("open", "high", "low", "close", "volume")
ADJUSTED_FIELDS = ("close",)
PRICE_COLUMNS = ("open", "high", "low", "close")
PROVIDER_COLUMNS = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)


def normalize_daily_frame(
    data: object,
    fields: Sequence[str] = RAW_FIELDS,
    price_divisor: float = FIINQUANT_PRICE_DIVISOR,
) -> pd.DataFrame:
    """Convert a FiinQuant trading-data response into a keyed daily frame."""

    if price_divisor <= 0:
        raise ValueError("Price divisor must be positive")

    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    if frame.empty:
        raise ValueError("FiinQuant returned no daily rows")

    normalized = frame.copy()
    normalized.columns = [
        str(column).strip().lower() for column in normalized.columns
    ]

    if "timestamp" in normalized.columns and "date" not in normalized.columns:
        normalized = normalized.rename(columns={"timestamp": "date"})

    required = {"date", "ticker", *fields}
    missing = sorted(required.difference(normalized.columns))

    if missing:
        raise ValueError(f"FiinQuant daily response is missing columns: {missing}")

    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="raise",
    ).dt.normalize()
    normalized["ticker"] = normalized["ticker"].astype(str).str.strip().str.upper()

    for column in fields:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if normalized[list(fields)].isna().any().any():
        raise ValueError("FiinQuant daily response contains nonnumeric values")

    for column in fields:
        if column in PRICE_COLUMNS:
            normalized[column] = normalized[column] / float(price_divisor)

    normalized = normalized[["date", "ticker", *fields]]

    if normalized.duplicated(["date", "ticker"]).any():
        raise ValueError("FiinQuant returned duplicate ticker-date rows")

    return normalized.sort_values(["ticker", "date"]).reset_index(drop=True)


class FiinQuantDailyProvider:
    """Serve per-ticker daily history from batched FiinQuant requests."""

    def __init__(
        self,
        client: object,
        symbols: Sequence[str],
        price_divisor: float = FIINQUANT_PRICE_DIVISOR,
    ) -> None:
        cleaned = [str(symbol).strip().upper() for symbol in symbols]

        if not cleaned or any(symbol == "" for symbol in cleaned):
            raise ValueError("FiinQuant provider requires non-empty symbols")

        self.client = client
        self.symbols = sorted(set(cleaned))
        self.price_divisor = float(price_divisor)
        self._window: tuple[date, date] | None = None
        self._frame: pd.DataFrame | None = None

    def history(
        self,
        ticker: str,
        provider_symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        frame = self._fetch_window(start_date, end_date)
        symbol = str(provider_symbol).strip().upper()
        rows = frame.loc[frame["ticker"] == symbol]

        if rows.empty:
            raise ValueError(
                f"FiinQuant returned no rows for {symbol} "
                f"between {start_date} and {end_date}"
            )

        return rows.drop(columns=["ticker"]).reset_index(drop=True)

    def _fetch(
        self,
        fields: Sequence[str],
        adjusted: bool,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        response = self.client.Fetch_Trading_Data(
            realtime=False,
            tickers=list(self.symbols),
            fields=list(fields),
            adjusted=adjusted,
            by="1d",
            from_date=start_date.isoformat(),
            to_date=end_date.isoformat(),
            lasted=False,
        ).get_data()

        return normalize_daily_frame(
            response,
            fields=fields,
            price_divisor=self.price_divisor,
        )

    def _fetch_window(self, start_date: date, end_date: date) -> pd.DataFrame:
        window = (start_date, end_date)

        # Only successful responses are cached, so the per-ticker retry loop in
        # download_update_rows still gets a fresh attempt after a failure.
        if self._frame is not None and self._window == window:
            return self._frame

        raw = self._fetch(
            fields=RAW_FIELDS,
            adjusted=False,
            start_date=start_date,
            end_date=end_date,
        )
        adjusted = self._fetch(
            fields=ADJUSTED_FIELDS,
            adjusted=True,
            start_date=start_date,
            end_date=end_date,
        ).rename(columns={"close": "adjusted_close"})

        combined = raw.merge(
            adjusted,
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
        )
        unmatched = combined.loc[combined["adjusted_close"].isna(), "ticker"]

        if not unmatched.empty:
            raise ValueError(
                "FiinQuant adjusted series does not cover every raw row: "
                f"{sorted(unmatched.unique())}"
            )

        frame = combined[list(PROVIDER_COLUMNS)]
        self._frame = frame
        self._window = window

        return frame
