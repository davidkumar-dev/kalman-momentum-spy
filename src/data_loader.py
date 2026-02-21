import pandas as pd
import os
from typing import Tuple


class DataLoader:

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.train_start = "2014-02-01"
        self.train_end = "2017-12-31"
        self.val_start = "2018-01-01"
        self.val_end = "2020-12-31"
        self.test_start = "2021-01-01"
        self.test_end = "2022-07-10"

    def load_spy_data(self, filename: str = "SPY_15min.csv") -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            alt = os.path.join(self.data_dir, "S&P-15-Min (1).csv")
            if os.path.exists(alt):
                filepath = alt
            else:
                raise FileNotFoundError(f"No data file found in {self.data_dir}")

        df = pd.read_csv(filepath)

        def parse_row(row):
            date_str = str(row["Date"]).strip()
            time_str = str(row["Time"]).strip()
            year_part = date_str.split("/")[-1]
            if len(year_part) == 2:
                fmt = "%m/%d/%y %I:%M:%S %p"
            else:
                fmt = "%d/%m/%Y %I:%M:%S %p"
            return pd.to_datetime(f"{date_str} {time_str}", format=fmt)

        df["datetime"] = df.apply(parse_row, axis=1)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        col_map = {}
        for c in df.columns:
            col_map[c] = c.lower()
        df.rename(columns=col_map, inplace=True)
        df.drop(columns=["date", "time"], errors="ignore", inplace=True)

        if df.index.year.min() < 2000:
            raise ValueError("Date parsing produced years before 2000")

        print(f"Loaded {len(df)} bars: {df.index[0]} to {df.index[-1]}")
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = df[self.train_start:self.train_end].copy()
        val = df[self.val_start:self.val_end].copy()
        test = df[self.test_start:self.test_end].copy()

        for name, subset in [("Train", train), ("Val", val), ("Test", test)]:
            print(f"  {name}: {subset.index[0]} to {subset.index[-1]} ({len(subset)} bars)")

        return train, val, test
