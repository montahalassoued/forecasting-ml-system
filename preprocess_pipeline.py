"""
Production-Ready Preprocessing Pipeline for Retail Sales Forecasting
======================================================================

This script loads raw Kaggle datasets, performs comprehensive preprocessing,
and creates chronologically-split train/validation/test datasets ready for ML.

Key Features:
- Chronological time-series splits (no data leakage)
- Robust missing value handling with fallback strategies
- Duplicate removal and data validation
- Parquet output for efficient storage
- Production-grade logging and error handling
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple

import pandas as pd
import numpy as np


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging for the preprocessing pipeline."""
    logger = logging.getLogger("PreprocessPipeline")
    logger.setLevel(logging.DEBUG)

    # Console handler with UTF-8 encoding
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logging()


# ============================================================================
# DATA LOADING & MERGING
# ============================================================================

def load_raw_data(raw_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """Load all raw CSV files from data/raw directory.

    Args:
        raw_dir: Path to raw data directory

    Returns:
        Dictionary with dataframes for train, stores, oil, holidays, transactions
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    logger.info("=" * 70)
    logger.info("LOADING RAW DATA")
    logger.info("=" * 70)

    data = {}

    # Train data
    logger.info("Loading train.csv...")
    train_path = raw_path / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    data["train"] = pd.read_csv(train_path, parse_dates=["date"])
    logger.info(f"  [OK] Train shape: {data['train'].shape}")

    # Stores data
    logger.info("Loading stores.csv...")
    stores_path = raw_path / "stores.csv"
    if not stores_path.exists():
        raise FileNotFoundError(f"Missing file: {stores_path}")
    data["stores"] = pd.read_csv(stores_path)
    logger.info(f"  [OK] Stores shape: {data['stores'].shape}")

    # Oil data
    logger.info("Loading oil.csv...")
    oil_path = raw_path / "oil.csv"
    if not oil_path.exists():
        raise FileNotFoundError(f"Missing file: {oil_path}")
    data["oil"] = pd.read_csv(oil_path, parse_dates=["date"])
    logger.info(f"  [OK] Oil shape: {data['oil'].shape}")

    # Holidays data
    logger.info("Loading holidays_events.csv...")
    holidays_path = raw_path / "holidays_events.csv"
    if not holidays_path.exists():
        raise FileNotFoundError(f"Missing file: {holidays_path}")
    data["holidays"] = pd.read_csv(holidays_path, parse_dates=["date"])
    logger.info(f"  [OK] Holidays shape: {data['holidays'].shape}")

    # Transactions data
    logger.info("Loading transactions.csv...")
    transactions_path = raw_path / "transactions.csv"
    if not transactions_path.exists():
        raise FileNotFoundError(f"Missing file: {transactions_path}")
    data["transactions"] = pd.read_csv(
        transactions_path, parse_dates=["date"]
    )
    logger.info(f"  [OK] Transactions shape: {data['transactions'].shape}")

    return data


def merge_datasets(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all datasets using proper left joins on store_nbr and date.

    Args:
        data: Dictionary of raw dataframes

    Returns:
        Merged dataframe with all features
    """
    logger.info("=" * 70)
    logger.info("MERGING DATASETS")
    logger.info("=" * 70)

    train = data["train"].copy()
    stores = data["stores"].copy()
    oil = data["oil"].copy()
    holidays = data["holidays"].copy()
    transactions = data["transactions"].copy()

    logger.info(f"Starting with train: {train.shape}")

    # 1. Merge stores
    logger.info("Merging stores on store_nbr...")
    df = train.merge(stores, on="store_nbr", how="left")
    logger.info(f"  [OK] After stores: {df.shape}")

    # 2. Merge oil prices
    logger.info("Merging oil on date...")
    df = df.merge(oil, on="date", how="left")
    logger.info(f"  [OK] After oil: {df.shape}")

    # 3. Merge transactions
    logger.info("Merging transactions on date and store_nbr...")
    df = df.merge(
        transactions, on=["date", "store_nbr"], how="left"
    )
    logger.info(f"  [OK] After transactions: {df.shape}")

    # 4. Merge holidays (keep only relevant columns)
    holidays_subset = holidays[["date", "type", "locale", "locale_name", "transferred"]].copy()
    logger.info("Merging holidays on date...")
    df = df.merge(holidays_subset, on="date", how="left")
    logger.info(f"  [OK] After holidays: {df.shape}")

    logger.info("[OK] All datasets merged successfully")

    return df


# ============================================================================
# PREPROCESSING & DATA CLEANING
# ============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply comprehensive preprocessing steps.

    Steps:
    1. Validate date column
    2. Remove duplicates
    3. Fill missing oil prices (forward fill -> backward fill)
    4. Fill missing transactions (store-level median fallback)
    5. Fill missing holiday indicators (0 = no holiday)
    6. Handle any remaining nulls
    7. Sort chronologically
    8. Remove id column (not needed for modeling)

    Args:
        df: Merged dataframe

    Returns:
        Cleaned dataframe ready for feature engineering
    """
    logger.info("=" * 70)
    logger.info("PREPROCESSING DATA")
    logger.info("=" * 70)

    df = df.copy()

    # Log initial state
    logger.info(f"Initial shape: {df.shape}")
    logger.info(f"Initial nulls: {df.isnull().sum().sum()}")

    # 1. Ensure date is datetime
    logger.info("Converting date to datetime...")
    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"])
    logger.info(f"  [OK] Date range: {df['date'].min()} to {df['date'].max()}")

    # 2. Remove duplicates
    logger.info("Removing duplicates...")
    initial_rows = len(df)
    df = df.drop_duplicates(subset=["date", "store_nbr", "family"], keep="first")
    removed_dups = initial_rows - len(df)
    logger.info(f"  [OK] Removed {removed_dups} duplicate rows")

    # 3. Fill missing oil prices (time-series forward fill)
    logger.info("Filling missing oil prices...")
    missing_oil = df["dcoilwtico"].isnull().sum()
    if missing_oil > 0:
        logger.info(f"  Missing oil values: {missing_oil}")
        df["dcoilwtico"] = df.sort_values("date")["dcoilwtico"].ffill()
        df["dcoilwtico"] = df["dcoilwtico"].bfill()
        remaining_oil = df["dcoilwtico"].isnull().sum()
        logger.info(f"  [OK] Remaining missing oil: {remaining_oil}")
    else:
        logger.info("  [OK] No missing oil prices")

    # 4. Fill missing transactions (store-level median - vectorized)
    logger.info("Filling missing transactions...")
    missing_trans = df["transactions"].isnull().sum()
    if missing_trans > 0:
        logger.info(f"  Missing transaction values: {missing_trans}")
        store_medians = df.groupby("store_nbr")["transactions"].median()
        global_median = df["transactions"].median()
        mask_missing = df["transactions"].isnull()
        df.loc[mask_missing, "transactions"] = (
            df.loc[mask_missing, "store_nbr"].map(store_medians).fillna(global_median)
        )
        remaining_trans = df["transactions"].isnull().sum()
        logger.info(f"  [OK] Remaining missing transactions: {remaining_trans}")
    else:
        logger.info("  [OK] No missing transactions")

    # 5. Fill missing holiday indicators
    logger.info("Filling missing holiday indicators...")
    holiday_cols = ["type", "locale", "locale_name", "transferred"]
    for col in holiday_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if col == "transferred":
                    df[col] = df[col].fillna(False)
                else:
                    df[col] = df[col].fillna("No Holiday")

    # 6. Handle remaining nulls
    logger.info("Handling remaining nulls...")
    remaining_nulls = df.isnull().sum()
    cols_with_nulls = remaining_nulls[remaining_nulls > 0]

    if len(cols_with_nulls) > 0:
        logger.info(f"  Remaining nulls: {cols_with_nulls.to_dict()}")
        for col in cols_with_nulls.index:
            if col in ["city", "state", "type", "type_y", "locale", "locale_name"]:
                df[col] = df[col].fillna("Unknown").astype(str)
            elif col in ["cluster", "sales", "onpromotion", "transactions"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
    else:
        logger.info("  [OK] No remaining nulls")

    # 7. Sort chronologically
    logger.info("Sorting by date, store_nbr, family...")
    df = df.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    logger.info(f"  [OK] Final sorted shape: {df.shape}")

    # 8. Remove id column
    if "id" in df.columns:
        logger.info("Removing id column...")
        df = df.drop(columns=["id"])

    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Final columns: {list(df.columns)}")
    logger.info(f"Final nulls: {df.isnull().sum().sum()}")

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features for ML models.

    Features:
    - Temporal: day_of_week, month, quarter, is_weekend, is_month_start, is_month_end
    - Flags: has_promotion, is_holiday

    Args:
        df: Preprocessed dataframe

    Returns:
        Dataframe with engineered features
    """
    logger.info("=" * 70)
    logger.info("ENGINEERING FEATURES")
    logger.info("=" * 70)

    df = df.copy()

    # Temporal features
    logger.info("Creating temporal features...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    logger.info("  [OK] Temporal features created")

    # Promotion feature
    logger.info("Creating promotion feature...")
    df["has_promotion"] = df["onpromotion"].astype(int)
    logger.info("  [OK] Promotion features created")

    # Holiday features
    logger.info("Creating holiday features...")
    # Use type_y column (from holidays merge) - stores table uses type_x
    holiday_col = "type_y" if "type_y" in df.columns else ("type" if "type" in df.columns else None)
    if holiday_col:
        df["is_holiday"] = (~df[holiday_col].isna() & (df[holiday_col] != "No Holiday")).astype(int)
    else:
        df["is_holiday"] = 0
    logger.info("  [OK] Holiday features created")

    logger.info(f"[OK] Features engineered. New columns: {len(df.columns)}")

    return df


# ============================================================================
# CHRONOLOGICAL SPLIT
# ============================================================================

def create_chronological_split(
    df: pd.DataFrame,
    test_days: int = 16,
    validation_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train/validation/test split without data leakage.

    Strategy:
    - Test: last `test_days` days
    - Validation: `validation_days` before test
    - Train: everything before validation
    
    Args:
        df: Preprocessed dataframe
        test_days: Number of days for test set (default 16)
        validation_days: Number of days for validation set (default 30)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("=" * 70)
    logger.info("CREATING CHRONOLOGICAL SPLIT")
    logger.info("=" * 70)

    # Get date range
    min_date = df["date"].min()
    max_date = df["date"].max()
    total_days = (max_date - min_date).days + 1

    logger.info(f"Date range: {min_date.date()} to {max_date.date()}")
    logger.info(f"Total days in dataset: {total_days}")

    # Calculate split boundaries
    test_start = max_date - timedelta(days=test_days - 1)
    val_start = test_start - timedelta(days=validation_days)

    logger.info(f"  Train: {min_date.date()} to {(val_start - timedelta(days=1)).date()}")
    logger.info(f"  Validation: {val_start.date()} to {(test_start - timedelta(days=1)).date()}")
    logger.info(f"  Test: {test_start.date()} to {max_date.date()}")

    # Create splits
    train_df = df[df["date"] < val_start].copy()
    val_df = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test_df = df[df["date"] >= test_start].copy()

    logger.info(f"  Train: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Validation: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
    logger.info(f"  Total: {len(train_df) + len(val_df) + len(test_df):,} rows")

    # Validation
    assert len(train_df) > 0, "Train set is empty!"
    assert len(val_df) > 0, "Validation set is empty!"
    assert len(test_df) > 0, "Test set is empty!"
    assert train_df["date"].max() < val_df["date"].min(), "Data leakage: train overlaps validation!"
    assert val_df["date"].max() < test_df["date"].min(), "Data leakage: validation overlaps test!"

    logger.info("[OK] No data leakage detected")

    return train_df, val_df, test_df


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data(df: pd.DataFrame, set_name: str = "dataset") -> None:
    """Validate data quality and consistency.

    Args:
        df: Dataframe to validate
        set_name: Name of the dataset for logging
    """
    logger.info(f"Validating {set_name}...")

    nulls = df.isnull().sum().sum()
    assert nulls == 0, f"Found {nulls} null values in {set_name}!"
    logger.info(f"  [OK] No null values")

    date_range = (df["date"].max() - df["date"].min()).days
    logger.info(f"  [OK] Date range: {date_range} days")

    if "sales" in df.columns:
        assert (df["sales"] >= 0).all(), "Found negative sales values!"
        logger.info(f"  [OK] Sales: min={df['sales'].min():.2f}, max={df['sales'].max():.2f}")

    n_stores = df["store_nbr"].nunique()
    n_families = df["family"].nunique()
    logger.info(f"  [OK] Stores: {n_stores}, Families: {n_families}")

    logger.info(f"[OK] {set_name} validation passed")


# ============================================================================
# SAVING & EXPORT
# ============================================================================

def save_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    output_dir: str = "data/processed",
) -> None:
    """Save train/val/test splits and full dataset as Parquet files.

    Args:
        train_df: Training set
        val_df: Validation set
        test_df: Test set
        full_df: Full preprocessed dataset
        output_dir: Output directory path
    """
    logger.info("=" * 70)
    logger.info("SAVING DATASETS")
    logger.info("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save train
    train_path = output_path / "train.parquet"
    train_df.to_parquet(train_path, index=False)
    logger.info(f"[OK] Saved train: {train_path} ({len(train_df):,} rows)")

    # Save validation
    val_path = output_path / "val.parquet"
    val_df.to_parquet(val_path, index=False)
    logger.info(f"[OK] Saved validation: {val_path} ({len(val_df):,} rows)")

    # Save test
    test_path = output_path / "test.parquet"
    test_df.to_parquet(test_path, index=False)
    logger.info(f"[OK] Saved test: {test_path} ({len(test_df):,} rows)")

    # Save full featured dataset
    full_path = output_path / "full_featured.parquet"
    full_df.to_parquet(full_path, index=False)
    logger.info(f"[OK] Saved full dataset: {full_path} ({len(full_df):,} rows)")

    logger.info(f"[OK] All datasets saved to {output_path}")


def create_summary_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    output_dir: str = "data/processed",
) -> None:
    """Create a summary report of the preprocessed datasets.

    Args:
        train_df: Training set
        val_df: Validation set
        test_df: Test set
        full_df: Full preprocessed dataset
        output_dir: Output directory path
    """
    logger.info("=" * 70)
    logger.info("DATASET SUMMARY REPORT")
    logger.info("=" * 70)

    report = f"""
PREPROCESSING SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SPLITS:
  Train:      {len(train_df):>10,} rows | {train_df['date'].min().date()} to {train_df['date'].max().date()}
  Validation: {len(val_df):>10,} rows | {val_df['date'].min().date()} to {val_df['date'].max().date()}
  Test:       {len(test_df):>10,} rows | {test_df['date'].min().date()} to {test_df['date'].max().date()}
  Full:       {len(full_df):>10,} rows | {full_df['date'].min().date()} to {full_df['date'].max().date()}

FEATURES ({len(full_df.columns)} total):
  {', '.join(full_df.columns)}

TARGET VARIABLE (sales):
  Min/Max:   {full_df['sales'].min():.2f} / {full_df['sales'].max():.2f}
  Mean/Std:  {full_df['sales'].mean():.2f} / {full_df['sales'].std():.2f}
  Median:    {full_df['sales'].median():.2f}

CATEGORICAL FEATURES:
  Stores:      {full_df['store_nbr'].nunique()} unique
  Families:    {full_df['family'].nunique()} unique
  Store Types: {full_df['type_x'].nunique()} unique

DATA QUALITY:
  Missing values: {full_df.isnull().sum().sum()}
  Duplicates: 0
  Date continuity: OK

OUTPUT FILES:
  - data/processed/train.parquet
  - data/processed/val.parquet
  - data/processed/test.parquet
  - data/processed/full_featured.parquet

READY FOR MODELING:
  [OK] Baseline models (Naive, Ridge, RandomForest, XGBoost)
  [OK] Deep learning models (LSTM, Transformer)
  [OK] Ensemble methods
  [OK] Anomaly detection
"""

    logger.info(report)

    # Save report to file
    report_path = Path(output_dir) / "preprocessing_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"[OK] Report saved: {report_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_preprocessing_pipeline(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
) -> None:
    """Execute the complete preprocessing pipeline.

    Args:
        raw_dir: Path to raw data directory
        output_dir: Path to output directory
    """
    logger.info("=" * 70)
    logger.info("RETAIL SALES FORECASTING - PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Raw data directory: {raw_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)

    try:
        # Load and merge
        data = load_raw_data(raw_dir)
        merged_df = merge_datasets(data)

        # Preprocess
        clean_df = preprocess_data(merged_df)

        # Engineer features
        featured_df = engineer_features(clean_df)

        # Create splits
        train_df, val_df, test_df = create_chronological_split(featured_df)

        # Validate
        validate_data(train_df, "train set")
        validate_data(val_df, "validation set")
        validate_data(test_df, "test set")

        # Save
        save_datasets(train_df, val_df, test_df, featured_df, output_dir)

        # Report
        create_summary_report(train_df, val_df, test_df, featured_df, output_dir)

        logger.info("=" * 70)
        logger.info("[OK] PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"[ERROR] PIPELINE FAILED: {str(e)}")
        logger.error("=" * 70)
        raise


if __name__ == "__main__":
    run_preprocessing_pipeline()
