
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd


RAW_FILENAMES: Mapping[str, str] = {
    "train": "train.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
}


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split configuration for the retail panel."""

    validation_days: int = 30
    test_days: int = 16
    date_col: str = "date"


@dataclass
class RetailPreprocessor:
    """Fit on train only, then transform the full chronological panel.

    The class stores train-derived statistics for robust outlier capping and fallback
    values for columns that may remain missing after past-only imputation.
    """

    sales_cap_quantile: float = 0.995
    transaction_cap_quantile: float = 0.995

    sales_caps_: pd.Series | None = None
    transaction_caps_: pd.Series | None = None
    transaction_fill_values_: pd.Series | None = None
    oil_fill_value_: float | None = None

    def fit(self, train_frame: pd.DataFrame) -> "RetailPreprocessor":
        required = {"family", "store_nbr", "sales", "transactions", "dcoilwtico"}
        missing = required.difference(train_frame.columns)
        if missing:
            raise ValueError(f"Missing required columns for fit: {sorted(missing)}")

        sales_source = train_frame[["family", "sales"]].dropna(subset=["family", "sales"])
        transaction_source = train_frame[["store_nbr", "transactions"]].dropna(
            subset=["store_nbr", "transactions"]
        )

        self.sales_caps_ = sales_source.groupby("family")["sales"].quantile(
            self.sales_cap_quantile
        )
        self.transaction_caps_ = transaction_source.groupby("store_nbr")["transactions"].quantile(
            self.transaction_cap_quantile
        )
        self.transaction_fill_values_ = transaction_source.groupby("store_nbr")["transactions"].median()
        oil_source = train_frame["dcoilwtico"].dropna()
        self.oil_fill_value_ = float(oil_source.median()) if not oil_source.empty else 0.0
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.sales_caps_ is None or self.transaction_caps_ is None:
            raise RuntimeError("RetailPreprocessor.fit must be called before transform.")

        processed = frame.copy()
        processed = processed.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)

        processed = self._fill_calendar_flags(processed)
        processed = self._fill_oil(processed)
        processed = self._fill_transactions(processed)
        processed = self._cap_target_sales(processed)

        if "onpromotion" in processed.columns:
            processed["onpromotion"] = (
                pd.to_numeric(processed["onpromotion"], errors="coerce")
                .fillna(0)
                .clip(lower=0)
                .astype("float64")
            )

        return processed

    def fit_transform(self, train_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_frame).transform(train_frame)

    def _fill_calendar_flags(self, frame: pd.DataFrame) -> pd.DataFrame:
        holiday_flag_columns = [
            column
            for column in [
                "is_holiday",
                "is_transfer_holiday",
                "is_bridge_holiday",
                "is_working_day",
                "is_national_holiday",
                "is_local_holiday",
                "is_regional_holiday",
            ]
            if column in frame.columns
        ]

        for column in holiday_flag_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int8")

        if "holiday_count" in frame.columns:
            frame["holiday_count"] = pd.to_numeric(frame["holiday_count"], errors="coerce").fillna(0)

        return frame

    def _fill_oil(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "dcoilwtico" not in frame.columns:
            return frame

        frame["dcoilwtico"] = pd.to_numeric(frame["dcoilwtico"], errors="coerce")
        frame["dcoilwtico"] = frame["dcoilwtico"].ffill()
        frame["dcoilwtico"] = frame["dcoilwtico"].fillna(self.oil_fill_value_)
        return frame

    def _fill_transactions(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "transactions" not in frame.columns:
            return frame

        frame["transactions"] = pd.to_numeric(frame["transactions"], errors="coerce")
        frame["transactions"] = frame.groupby("store_nbr", sort=False)["transactions"].transform(
            lambda series: series.ffill()
        )

        if self.transaction_fill_values_ is not None:
            fallback = frame["store_nbr"].map(self.transaction_fill_values_)
            frame["transactions"] = frame["transactions"].fillna(fallback)

        frame["transactions"] = frame["transactions"].fillna(frame["transactions"].median())
        frame["transactions"] = frame["transactions"].clip(lower=0)

        cap_map = self.transaction_caps_
        if cap_map is not None:
            frame["transactions"] = frame["transactions"].clip(upper=frame["store_nbr"].map(cap_map))

        return frame

    def _cap_target_sales(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "sales" not in frame.columns:
            return frame

        frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
        cap_map = self.sales_caps_
        if cap_map is not None:
            frame["sales"] = frame["sales"].clip(lower=0, upper=frame["family"].map(cap_map))
        else:
            frame["sales"] = frame["sales"].clip(lower=0)
        return frame


def load_raw_datasets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load the raw CSV files required for the retail panel."""

    data_dir = Path(data_dir)
    frames: Dict[str, pd.DataFrame] = {}

    for name, filename in RAW_FILENAMES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw file: {path}")

        parse_dates = ["date"] if "date" in pd.read_csv(path, nrows=0).columns else None
        if parse_dates:
            frames[name] = pd.read_csv(path, parse_dates=parse_dates)
        else:
            frames[name] = pd.read_csv(path)

    return frames


def build_holiday_calendar(holidays: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw holiday table into daily binary calendar flags."""

    if holidays.empty:
        return pd.DataFrame(columns=["date", "holiday_count", "is_holiday", "is_transfer_holiday"])

    calendar = holidays.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["holiday_type"] = calendar.get("type", pd.Series(index=calendar.index, dtype="object")).astype(str)
    calendar["locale"] = calendar.get("locale", pd.Series(index=calendar.index, dtype="object")).astype(str)
    calendar["transferred"] = (
        calendar.get("transferred", pd.Series(index=calendar.index, dtype="object"))
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )

    grouped = calendar.groupby("date", as_index=False)
    daily = grouped.agg(
        holiday_count=("holiday_type", "size"),
        is_holiday=("holiday_type", lambda values: int((values == "Holiday").any())),
        is_transfer_holiday=("transferred", "max"),
        is_bridge_holiday=("holiday_type", lambda values: int((values == "Bridge").any())),
        is_working_day=("holiday_type", lambda values: int((values == "Work Day").any())),
        is_national_holiday=("locale", lambda values: int((values == "National").any())),
        is_local_holiday=("locale", lambda values: int((values == "Local").any())),
        is_regional_holiday=("locale", lambda values: int((values == "Regional").any())),
    )

    return daily


def merge_sources(
    train: pd.DataFrame,
    stores: pd.DataFrame,
    transactions: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full store-family-date panel with daily exogenous signals."""

    sales_panel = train.copy()
    sales_panel["date"] = pd.to_datetime(sales_panel["date"])

    stores_frame = stores.copy()
    stores_frame = stores_frame.drop_duplicates(subset=["store_nbr"])

    panel = sales_panel.merge(stores_frame, on="store_nbr", how="left", validate="m:1")

    daily_index = panel[["date", "store_nbr"]].drop_duplicates().sort_values(["store_nbr", "date"])

    transactions_daily = transactions.copy()
    transactions_daily["date"] = pd.to_datetime(transactions_daily["date"])
    transactions_daily = (
        transactions_daily.groupby(["store_nbr", "date"], as_index=False)["transactions"].sum().sort_values(
            ["store_nbr", "date"]
        )
    )

    oil_daily = oil.copy()
    oil_daily["date"] = pd.to_datetime(oil_daily["date"])
    oil_daily = oil_daily[["date", "dcoilwtico"]].drop_duplicates(subset=["date"]).sort_values("date")
    oil_daily = oil_daily.set_index("date").asfreq("D")
    oil_daily["dcoilwtico"] = pd.to_numeric(oil_daily["dcoilwtico"], errors="coerce").ffill()
    oil_daily = oil_daily.reset_index()

    holiday_calendar = build_holiday_calendar(holidays)

    exogenous = daily_index.merge(transactions_daily, on=["store_nbr", "date"], how="left")
    exogenous = exogenous.merge(oil_daily, on="date", how="left")
    exogenous = exogenous.merge(holiday_calendar, on="date", how="left")

    return panel.merge(exogenous, on=["store_nbr", "date"], how="left", validate="m:1")


def split_by_time(
    frame: pd.DataFrame,
    split_config: SplitConfig = SplitConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a chronological train/validation/test split.

    The split is computed from unique dates, so every store-family row for a given
    day stays in the same partition.
    """

    if split_config.date_col not in frame.columns:
        raise ValueError(f"Missing date column: {split_config.date_col}")

    unique_dates = pd.Index(pd.to_datetime(frame[split_config.date_col]).dropna().unique()).sort_values()
    if len(unique_dates) <= split_config.validation_days + split_config.test_days:
        raise ValueError("Not enough dates to create the requested validation/test split.")

    test_dates = unique_dates[-split_config.test_days :]
    validation_dates = unique_dates[-(split_config.validation_days + split_config.test_days) : -split_config.test_days]
    train_dates = unique_dates[: -(split_config.validation_days + split_config.test_days)]

    train_mask = frame[split_config.date_col].isin(train_dates)
    validation_mask = frame[split_config.date_col].isin(validation_dates)
    test_mask = frame[split_config.date_col].isin(test_dates)

    return frame[train_mask].copy(), frame[validation_mask].copy(), frame[test_mask].copy()


def prepare_datasets(
    data_dir: Path,
    split_config: SplitConfig = SplitConfig(),
) -> dict[str, pd.DataFrame | RetailPreprocessor]:
    """Load, merge, clean and split the retail panel in a leakage-safe way."""

    raw = load_raw_datasets(data_dir)
    merged = merge_sources(
        raw["train"],
        raw["stores"],
        raw["transactions"],
        raw["oil"],
        raw["holidays"],
    )

    train_frame, validation_frame, test_frame = split_by_time(merged, split_config)

    preprocessor = RetailPreprocessor().fit(train_frame)
    processed = preprocessor.transform(merged)

    train_frame, validation_frame, test_frame = split_by_time(processed, split_config)

    return {
        "full": processed,
        "train": train_frame,
        "validation": validation_frame,
        "test": test_frame,
        "preprocessor": preprocessor,
    }