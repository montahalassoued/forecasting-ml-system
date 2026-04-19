
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
	"""Configuration for time-series feature generation."""

	date_col: str = "date"
	group_cols: tuple[str, ...] = ("store_nbr", "family")
	target_col: str = "sales"
	transaction_col: str = "transactions"
	promotion_col: str = "onpromotion"
	oil_col: str = "dcoilwtico"
	lags: tuple[int, ...] = (1, 7, 14, 28)
	rolling_windows: tuple[int, ...] = (7, 14, 28)
	drop_incomplete_rows: bool = True


def _ensure_date(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
	if date_col not in frame.columns:
		raise ValueError(f"Missing required date column: {date_col}")
	result = frame.copy()
	result[date_col] = pd.to_datetime(result[date_col])
	return result


def _safe_series(frame: pd.DataFrame, column: str) -> pd.Series | None:
	return frame[column] if column in frame.columns else None


def _group_shift(frame: pd.DataFrame, group_cols: Sequence[str], column: str, lag: int) -> pd.Series:
	return frame.groupby(list(group_cols), sort=False)[column].shift(lag)


def _group_shift_rolling(
	frame: pd.DataFrame,
	group_cols: Sequence[str],
	column: str,
	window: int,
	agg: str,
) -> pd.Series:
	shifted = frame.groupby(list(group_cols), sort=False)[column].shift(1)
	grouped = shifted.groupby([frame[col] for col in group_cols], sort=False)

	if agg == "mean":
		return grouped.transform(lambda series: series.rolling(window=window, min_periods=1).mean())
	if agg == "std":
		return grouped.transform(lambda series: series.rolling(window=window, min_periods=2).std()).fillna(0.0)
	if agg == "min":
		return grouped.transform(lambda series: series.rolling(window=window, min_periods=1).min())
	if agg == "max":
		return grouped.transform(lambda series: series.rolling(window=window, min_periods=1).max())

	raise ValueError(f"Unsupported rolling aggregation: {agg}")


def _add_calendar_features(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
	date = frame[date_col]

	frame["day"] = date.dt.day
	frame["dayofweek"] = date.dt.dayofweek
	frame["weekofyear"] = date.dt.isocalendar().week.astype("int16")
	frame["month"] = date.dt.month
	frame["quarter"] = date.dt.quarter
	frame["year"] = date.dt.year
	frame["dayofyear"] = date.dt.dayofyear
	frame["is_weekend"] = date.dt.dayofweek.isin([5, 6]).astype("int8")
	frame["is_month_start"] = date.dt.is_month_start.astype("int8")
	frame["is_month_end"] = date.dt.is_month_end.astype("int8")

	frame["dayofweek_sin"] = np.sin(2.0 * pi * frame["dayofweek"] / 7.0)
	frame["dayofweek_cos"] = np.cos(2.0 * pi * frame["dayofweek"] / 7.0)
	frame["month_sin"] = np.sin(2.0 * pi * (frame["month"] - 1) / 12.0)
	frame["month_cos"] = np.cos(2.0 * pi * (frame["month"] - 1) / 12.0)
	frame["dayofyear_sin"] = np.sin(2.0 * pi * (frame["dayofyear"] - 1) / 365.25)
	frame["dayofyear_cos"] = np.cos(2.0 * pi * (frame["dayofyear"] - 1) / 365.25)

	return frame


def _add_holiday_features(frame: pd.DataFrame) -> pd.DataFrame:
	holiday_columns = [
		"holiday_count",
		"is_holiday",
		"is_transfer_holiday",
		"is_bridge_holiday",
		"is_working_day",
		"is_national_holiday",
		"is_local_holiday",
		"is_regional_holiday",
	]

	for column in holiday_columns:
		if column in frame.columns:
			frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int8")

	return frame


def _add_lag_block(
	frame: pd.DataFrame,
	group_cols: Sequence[str],
	column: str,
	lags: Sequence[int],
	prefix: str | None = None,
) -> pd.DataFrame:
	if column not in frame.columns:
		return frame

	feature_prefix = prefix or column
	for lag in lags:
		frame[f"{feature_prefix}_lag_{lag}"] = _group_shift(frame, group_cols, column, lag)

	return frame


def _add_rolling_block(
	frame: pd.DataFrame,
	group_cols: Sequence[str],
	column: str,
	windows: Sequence[int],
	prefix: str | None = None,
) -> pd.DataFrame:
	if column not in frame.columns:
		return frame

	feature_prefix = prefix or column
	for window in windows:
		frame[f"{feature_prefix}_roll_mean_{window}"] = _group_shift_rolling(frame, group_cols, column, window, "mean")
		frame[f"{feature_prefix}_roll_std_{window}"] = _group_shift_rolling(frame, group_cols, column, window, "std")
		frame[f"{feature_prefix}_roll_min_{window}"] = _group_shift_rolling(frame, group_cols, column, window, "min")
		frame[f"{feature_prefix}_roll_max_{window}"] = _group_shift_rolling(frame, group_cols, column, window, "max")

	return frame


def build_features(frame: pd.DataFrame, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
	"""Create a model-ready feature matrix without leaking future information."""

	enriched = _ensure_date(frame, config.date_col)
	sort_columns = [config.date_col, *config.group_cols]
	enriched = enriched.sort_values(sort_columns).reset_index(drop=True)

	enriched = _add_calendar_features(enriched, config.date_col)
	enriched = _add_holiday_features(enriched)

	if config.promotion_col in enriched.columns:
		enriched[config.promotion_col] = pd.to_numeric(enriched[config.promotion_col], errors="coerce").fillna(0)
		enriched["promo_active"] = (enriched[config.promotion_col] > 0).astype("int8")

	# Core demand history
	enriched = _add_lag_block(enriched, config.group_cols, config.target_col, config.lags, prefix="sales")
	enriched = _add_rolling_block(enriched, config.group_cols, config.target_col, config.rolling_windows, prefix="sales")

	# Transactions are highly correlated with sales, but only safe when lagged.
	enriched = _add_lag_block(
		enriched,
		config.group_cols,
		config.transaction_col,
		config.lags,
		prefix="transactions",
	)
	enriched = _add_rolling_block(
		enriched,
		config.group_cols,
		config.transaction_col,
		config.rolling_windows,
		prefix="transactions",
	)

	# Promotion history can be useful because the promotion calendar is often known in advance.
	enriched = _add_lag_block(
		enriched,
		config.group_cols,
		config.promotion_col,
		config.lags,
		prefix="promo",
	)
	enriched = _add_rolling_block(
		enriched,
		config.group_cols,
		config.promotion_col,
		config.rolling_windows,
		prefix="promo",
	)

	# Oil is weakly related, so we keep it only as a lagged exogenous signal.
	enriched = _add_lag_block(enriched, config.group_cols, config.oil_col, config.lags, prefix="oil")
	enriched = _add_rolling_block(enriched, config.group_cols, config.oil_col, config.rolling_windows, prefix="oil")

	# Useful stability feature for retail time-series models.
	enriched["days_since_start"] = (
		enriched.groupby(list(config.group_cols), sort=False).cumcount().astype("int32")
	)

	if config.drop_incomplete_rows:
		feature_columns = [
			column
			for column in enriched.columns
			if any(
				token in column
				for token in [
					"_lag_",
					"_roll_mean_",
					"_roll_std_",
					"_roll_min_",
					"_roll_max_",
				]
			)
		]
		enriched = enriched.dropna(subset=feature_columns).reset_index(drop=True)

	return enriched


def split_feature_frame(
	frame: pd.DataFrame,
	split_config: tuple[pd.Timestamp | str, pd.Timestamp | str],
	date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Split an already-featured frame using date boundaries.

	The tuple must contain (validation_start, test_start). Everything before the
	validation start is training data, values in [validation_start, test_start) are
	validation data, and values from test_start onward are test data.
	"""

	validation_start, test_start = pd.to_datetime(split_config[0]), pd.to_datetime(split_config[1])
	dates = pd.to_datetime(frame[date_col])

	train_mask = dates < validation_start
	validation_mask = (dates >= validation_start) & (dates < test_start)
	test_mask = dates >= test_start

	return frame[train_mask].copy(), frame[validation_mask].copy(), frame[test_mask].copy()
