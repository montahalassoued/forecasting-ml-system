# Preprocessing Pipeline - Usage Guide

## Overview

The `preprocess_pipeline.py` script transforms raw Kaggle retail sales data into production-ready datasets for machine learning models, with proper chronological splitting to prevent data leakage.

## Quick Start

```bash
python preprocess_pipeline.py
```

The script will:

1. Load all raw CSV files from `data/raw/`
2. Merge datasets intelligently
3. Clean and preprocess all values
4. Engineer time-based features
5. Create chronological train/validation/test splits
6. Save 4 parquet files in `data/processed/`

## Output Files

### Generated Parquet Files

All files are located in `data/processed/`:

| File                    | Rows      | Purpose                                           |
| ----------------------- | --------- | ------------------------------------------------- |
| `train.parquet`         | 2,918,916 | 97.3% - Training data (2013-01-01 to 2017-06-30)  |
| `val.parquet`           | 53,460    | 1.8% - Validation data (2017-07-01 to 2017-07-30) |
| `test.parquet`          | 28,512    | 1.0% - Test data (2017-07-31 to 2017-08-15)       |
| `full_featured.parquet` | 3,000,888 | All data with engineered features                 |

### Dataset Structure

All files have **23 columns**:

**Temporal Columns:**

- `date` - Transaction date (datetime)
- `day_of_week` - 0-6 (Monday-Sunday)
- `month` - 1-12
- `quarter` - 1-4
- `is_weekend` - Binary (0/1)
- `is_month_start` - Binary (0/1)
- `is_month_end` - Binary (0/1)

**Store & Inventory Features:**

- `store_nbr` - Store identifier (1-54)
- `family` - Product family (33 categories)
- `city` - Store location city
- `state` - Store location state
- `type_x` - Store type (A, B, C, D, E)
- `cluster` - Store cluster (1-17)

**Sales & Transaction Data:**

- `sales` - Target variable (sales amount)
- `onpromotion` - Count of items on promotion
- `has_promotion` - Binary promotion flag
- `transactions` - Number of transactions
- `dcoilwtico` - Oil price (Brent Crude)

**Holiday & Event Features:**

- `type_y` - Holiday type (Holiday, Bridge, Event, etc.)
- `locale` - Holiday scope (National, Regional, Local)
- `locale_name` - Specific region name
- `transferred` - Whether holiday was transferred
- `is_holiday` - Binary holiday flag

## Data Quality

✓ **0 missing values** - All nulls handled intelligently
✓ **No duplicates** - 53,460 duplicates removed
✓ **Chronologically sorted** - By date, store_nbr, family
✓ **No data leakage** - Test set strictly after validation

## Preprocessing Details

### Merges Performed

1. **Stores** - Location and store metadata (1-to-many)
2. **Oil Prices** - Daily Brent crude (date join)
3. **Transactions** - Daily store transactions
4. **Holidays** - National, regional, local events

### Missing Value Handling

- **Oil prices**: Forward fill + backward fill (time-series imputation)
- **Transactions**: Store-level median fill + global median fallback
- **Holidays**: Zero-filled or "No Holiday" marker
- **Store metadata**: "Unknown" fill for categorical

### Feature Engineering

- Temporal features extracted from date column
- Promotion binary indicator created
- Holiday binary indicator (any non-"No Holiday" event)

## Using with Kaggle Models

### Load & Train with Baseline Model

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_parquet('data/processed/train.parquet')
val = pd.read_parquet('data/processed/val.parquet')
test = pd.read_parquet('data/processed/test.parquet')

# Encode categorical columns
le_family = LabelEncoder()
le_store = LabelEncoder()
le_city = LabelEncoder()

train['family_encoded'] = le_family.fit_transform(train['family'])
train['store_encoded'] = le_store.fit_transform(train['store_nbr'])
train['city_encoded'] = le_city.fit_transform(train['city'])

# Select features
feature_cols = [
    'store_encoded', 'family_encoded', 'city_encoded', 'cluster',
    'dcoilwtico', 'transactions', 'onpromotion',
    'day_of_week', 'month', 'quarter', 'is_weekend',
    'is_month_start', 'is_month_end', 'has_promotion', 'is_holiday'
]

X_train = train[feature_cols]
y_train = train['sales']

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict on validation
val_encoded = val.copy()
val_encoded['family_encoded'] = le_family.transform(val_encoded['family'])
val_encoded['store_encoded'] = le_store.transform(val_encoded['store_nbr'])
val_encoded['city_encoded'] = le_city.transform(val_encoded['city'])

val_pred = model.predict(val_encoded[feature_cols])
```

### Models Ready to Use

These datasets are optimized for:

- **Baseline Models**: Naive forecasting, Ridge regression, RandomForest, XGBoost
- **Time Series Models**: LSTM, Temporal Convolutional Networks, Transformer
- **Ensemble Methods**: Stacking, Blending, Voting
- **Anomaly Detection**: Autoencoders, IsolationForest
- **Advanced Models**: Prophet, ARIMA/SARIMA, Seasonal decomposition

## Preventing Data Leakage

The splits are strictly chronological:

```
Train Set:       [2013-01-01 ..................... 2017-06-30]  97.3% of data
Validation Set:                              [2017-07-01 ... 2017-07-30]  1.8% of data
Test Set:                                    [2017-07-31 ... 2017-08-15]  1.0% of data

Zero Overlap - Each set uses only past/concurrent data for training
```

**Key safeguards:**

- Train only sees data up to 2017-06-30
- Validation never touches training data
- Test never sees train or validation data

## Performance Tips for Kaggle

1. **Use all features** - They're all engineered and validated
2. **Scale numerical features** - Especially `dcoilwtico` and `transactions`
3. **One-hot encode** - `family`, `city`, `type_x`, `locale`
4. **Use validation metrics** - RMSE, MAE, MAPE on val.parquet
5. **Ensemble predictions** - Combine multiple model predictions
6. **Time-aware CV** - Use temporal cross-validation, not random
7. **Feature importance** - Analyze store/family/temporal patterns

## Troubleshooting

### Memory Issues

- If memory is insufficient, process splits separately:
  ```python
  train = pd.read_parquet('data/processed/train.parquet',
                         columns=['store_nbr', 'family', 'sales', ...])
  ```

### Parquet Not Installed

```bash
pip install pyarrow
```

### Missing Data Files

Ensure all 5 raw files exist in `data/raw/`:

- `train.csv` (3M rows)
- `stores.csv` (54 rows)
- `oil.csv` (1,218 rows)
- `holidays_events.csv` (350 rows)
- `transactions.csv` (83,488 rows)

## Running on Kaggle Notebook

```python
# In Kaggle Notebook, run this to load preprocessed data
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

train = pd.read_parquet('/kaggle/working/train.parquet')  # After preprocessing
val = pd.read_parquet('/kaggle/working/val.parquet')
test = pd.read_parquet('/kaggle/working/test.parquet')

print(f"Train: {train.shape}")
print(f"Val: {val.shape}")
print(f"Test: {test.shape}")
print(f"\nColumns: {train.columns.tolist()}")
print(f"\nTarget stats:\n{train['sales'].describe()}")
```

## Success Checklist

Before submitting to Kaggle:

- [ ] All 4 parquet files generated (4 files in data/processed/)
- [ ] No null values in any file (check `Final nulls: 0` in log)
- [ ] Train/val/test chronologically ordered (no future leakage)
- [ ] 23 features available for modeling
- [ ] Sales target ranges 0.00 to 124,717.00
- [ ] Baseline model trained successfully on train.parquet
- [ ] Validation metrics computed on val.parquet

---

**Last Run**: 2026-04-20
**Data Range**: 2013-01-01 to 2017-08-15
**Records**: 3,000,888 transactions
