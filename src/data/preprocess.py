import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import numpy as np
from pathlib import Path

def load_and_merge(data_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(data_dir / 'train.csv', parse_dates=['date'])
    stores = pd.read_csv(data_dir / 'stores.csv')
    oil = pd.read_csv(data_dir / 'oil.csv', parse_dates=['date'])
    holidays_events = pd.read_csv(data_dir / 'holidays_events.csv', parse_dates=['date'])
    
    df = train.merge(stores, on='store_nbr', how='left')
    df = df.merge(oil, on='date', how='left')
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Fill oil price gaps (weekends have no price)
    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()
    
    # Log-transform sales to stabilize variance (common in retail)
    # log1p because sales can be 0
    df['sales_log'] = np.log1p(df['sales'])
    
    # Clip extreme outliers at 99.5th percentile per family
    p995 = df.groupby('family')['sales'].transform(
        lambda x: x.quantile(0.995)
    )
    df['sales_clipped'] = df['sales'].clip(upper=p995)
    
    return df

def time_split(df: pd.DataFrame, val_days: int = 30, test_days: int = 16):
    """
    Walk-forward split — NEVER use random split on time series.
    Leaking future data into training is the #1 mistake in time series ML.
    """
    max_date = df['date'].max()
    test_start = max_date - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    
    train = df[df['date'] < val_start]
    val   = df[(df['date'] >= val_start) & (df['date'] < test_start)]
    test  = df[df['date'] >= test_start]
    
    return train, val, test