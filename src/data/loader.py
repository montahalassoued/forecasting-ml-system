from pathlib import Path
import pandas as pd


class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_csv(self, filename: str, parse_dates=None):
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.read_csv(path, parse_dates=parse_dates)

    def load_all(self):
        train = self.load_csv("train.csv", parse_dates=["date"])
        stores = self.load_csv("stores.csv")
        oil = self.load_csv("oil.csv", parse_dates=["date"])
        holidays = self.load_csv("holidays_events.csv", parse_dates=["date"])
        transactions = self.load_csv("transactions.csv", parse_dates=["date"])

        return {
            "train": train,
            "stores": stores,
            "oil": oil,
            "holidays": holidays,
            "transactions": transactions
        }


def merge_data(data: dict) -> pd.DataFrame:
    train = data["train"]
    stores = data["stores"]
    oil = data["oil"]
    holidays = data["holidays"]
    transactions = data["transactions"]

    # 1. merge stores
    df = train.merge(stores, on="store_nbr", how="left")

    # 2. merge oil
    df = df.merge(oil, on="date", how="left")

    # 3. merge transactions
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")

    # 4. holidays → join by date
    df = df.merge(holidays, on="date", how="left")

    return df


def load_pipeline(data_dir: str):
    loader = DataLoader(data_dir)
    data = loader.load_all()
    df = merge_data(data)

    return df