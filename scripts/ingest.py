import os
from pathlib import Path
import pandas as pd
from scripts.db_utils import get_engine

REQUIRED_COLUMNS = [
    "Transaction_ID",
    "Transaction_date",
    "Gender",
    "Age",
    "Marital_status",
    "State_names",
    "Segment",
    "Employees_status",
    "Payment_method",
    "Referral",
    "Amount_spent",
]

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

def load_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        sample_path = Path("/app/data/raw/sample_transactions.csv")
        if sample_path.exists():
            print("WARNING: data/raw/transactions.csv was not found.")
            print("Using data/raw/sample_transactions.csv only for a small test run.")
            csv_path = sample_path
        else:
            raise FileNotFoundError(
                "Dataset not found. Place the full Kaggle CSV at data/raw/transactions.csv"
            )

    df = pd.read_csv(csv_path)
    df = clean_column_names(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    print(f"Loaded dataset from {csv_path}")
    print(f"Rows: {len(df):,}, Columns: {len(df.columns)}")
    return df

def main():
    data_file = os.getenv("DATA_FILE", "/app/data/raw/transactions.csv")
    df = load_csv(data_file)

    engine = get_engine()
    df.to_sql("raw_transactions", engine, if_exists="replace", index=False, chunksize=10000, method="multi")

    print("Ingestion completed successfully.")
    print("Table created: raw_transactions")

if __name__ == "__main__":
    main()
