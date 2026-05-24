import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.db_utils import get_engine

PROCESSED_DIR = Path("/app/data/processed")
OUTPUT_DIR = Path("/app/data/output")
MODELS_DIR = Path("/app/models")

for directory in [PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def save_plot(filename: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    df = df.drop_duplicates()
    df["Transaction_date"] = pd.to_datetime(df["Transaction_date"], errors="coerce")
    df["Amount_spent"] = pd.to_numeric(df["Amount_spent"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df = df.dropna(subset=["Transaction_ID", "Transaction_date", "Amount_spent", "Age"])

    categorical_cols = [
        "Gender",
        "Marital_status",
        "State_names",
        "Segment",
        "Employees_status",
        "Payment_method",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["Referral"] = pd.to_numeric(df["Referral"], errors="coerce").fillna(0).astype(int)

    df["Year"] = df["Transaction_date"].dt.year
    df["Month"] = df["Transaction_date"].dt.month
    df["Quarter"] = df["Transaction_date"].dt.quarter
    df["Day"] = df["Transaction_date"].dt.day
    df["Hour"] = df["Transaction_date"].dt.hour
    df["Month_Year"] = df["Transaction_date"].dt.to_period("M").astype(str)

    return df

def create_aggregations(df: pd.DataFrame):
    monthly_sales = (
        df.groupby("Month_Year")
        .agg(
            total_sales=("Amount_spent", "sum"),
            average_sales=("Amount_spent", "mean"),
            transaction_count=("Transaction_ID", "count"),
        )
        .reset_index()
    )

    quarterly_sales = (
        df.groupby(["Year", "Quarter"])
        .agg(
            total_sales=("Amount_spent", "sum"),
            average_sales=("Amount_spent", "mean"),
            transaction_count=("Transaction_ID", "count"),
        )
        .reset_index()
    )

    segment_sales = (
        df.groupby("Segment")
        .agg(
            total_sales=("Amount_spent", "sum"),
            average_sales=("Amount_spent", "mean"),
            transaction_count=("Transaction_ID", "count"),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )

    state_sales = (
        df.groupby("State_names")
        .agg(
            total_sales=("Amount_spent", "sum"),
            average_sales=("Amount_spent", "mean"),
            transaction_count=("Transaction_ID", "count"),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )

    payment_sales = (
        df.groupby("Payment_method")
        .agg(total_sales=("Amount_spent", "sum"))
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )

    return monthly_sales, quarterly_sales, segment_sales, state_sales, payment_sales

def create_visualizations(monthly_sales, segment_sales, state_sales, payment_sales):
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_sales["Month_Year"], monthly_sales["total_sales"])
    plt.title("Monthly Total Sales")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=90)
    save_plot("monthly_total_sales.png")

    plt.figure(figsize=(14, 6))
    plt.plot(monthly_sales["Month_Year"], monthly_sales["average_sales"])
    plt.title("Monthly Average Sales")
    plt.xlabel("Month")
    plt.ylabel("Average Sales")
    plt.xticks(rotation=90)
    save_plot("monthly_average_sales.png")

    plt.figure(figsize=(14, 6))
    plt.bar(monthly_sales["Month_Year"], monthly_sales["transaction_count"])
    plt.title("Monthly Transaction Count")
    plt.xlabel("Month")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=90)
    save_plot("monthly_transaction_count.png")

    monthly_sales = monthly_sales.copy()
    monthly_sales["rolling_avg"] = monthly_sales["total_sales"].rolling(window=3).mean()
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_sales["Month_Year"], monthly_sales["total_sales"], label="Actual Sales")
    plt.plot(monthly_sales["Month_Year"], monthly_sales["rolling_avg"], label="3-Month Moving Average")
    plt.title("Monthly Sales Trend with Moving Average")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.xticks(rotation=90)
    save_plot("monthly_sales_moving_average.png")

    plt.figure(figsize=(9, 5))
    plt.bar(segment_sales["Segment"], segment_sales["total_sales"])
    plt.title("Total Sales by Customer Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Total Sales")
    save_plot("segment_sales.png")

    plt.figure(figsize=(9, 5))
    plt.bar(payment_sales["Payment_method"], payment_sales["total_sales"])
    plt.title("Total Sales by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Total Sales")
    save_plot("payment_method_sales.png")

    top_states = state_sales.head(10)
    plt.figure(figsize=(12, 6))
    plt.bar(top_states["State_names"], top_states["total_sales"])
    plt.title("Top 10 States by Total Sales")
    plt.xlabel("State")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    save_plot("top_10_states_sales.png")

def train_model(df: pd.DataFrame):
    features = [
        "Gender",
        "Age",
        "Marital_status",
        "State_names",
        "Segment",
        "Employees_status",
        "Payment_method",
        "Referral",
        "Year",
        "Month",
        "Quarter",
        "Hour",
    ]
    target = "Amount_spent"

    model_df = df[features + [target]].dropna()

    max_rows = int(os.getenv("ML_MAX_ROWS", "100000"))
    if len(model_df) > max_rows:
        model_df = model_df.sample(n=max_rows, random_state=42)
        print(f"Model training sampled to {max_rows:,} rows for local Docker performance.")

    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = [
        "Gender",
        "Marital_status",
        "State_names",
        "Segment",
        "Employees_status",
        "Payment_method",
    ]
    numeric_features = ["Age", "Referral", "Year", "Month", "Quarter", "Hour"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=18,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results = pd.DataFrame(
        {
            "Actual_Amount_Spent": y_test.values,
            "Predicted_Amount_Spent": y_pred,
        }
    )
    results.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)
    joblib.dump(pipeline, MODELS_DIR / "amount_spent_prediction_model.pkl")

    sample_results = results.head(100)
    plt.figure(figsize=(12, 6))
    plt.plot(sample_results["Actual_Amount_Spent"].values, label="Actual")
    plt.plot(sample_results["Predicted_Amount_Spent"].values, label="Predicted")
    plt.title("Actual vs Predicted Amount Spent")
    plt.xlabel("Sample Records")
    plt.ylabel("Amount Spent")
    plt.legend()
    save_plot("actual_vs_predicted.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title("Actual vs Predicted Scatter Plot")
    plt.xlabel("Actual Amount")
    plt.ylabel("Predicted Amount")
    save_plot("scatter_actual_vs_predicted.png")

    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50)
    plt.title("Residual Error Distribution")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    save_plot("residual_error_distribution.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0)
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    save_plot("residual_plot.png")

    return {
        "model_rows": len(model_df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

def main():
    engine = get_engine()

    print("Reading raw_transactions table from PostgreSQL.")
    df = pd.read_sql("SELECT * FROM raw_transactions", engine)

    print("Preprocessing data.")
    cleaned = preprocess(df)
    cleaned.to_csv(PROCESSED_DIR / "cleaned_transactions.csv", index=False)
    cleaned.to_sql("processed_transactions", engine, if_exists="replace", index=False, chunksize=10000, method="multi")

    print("Creating aggregations.")
    monthly_sales, quarterly_sales, segment_sales, state_sales, payment_sales = create_aggregations(cleaned)

    outputs = {
        "monthly_sales": monthly_sales,
        "quarterly_sales": quarterly_sales,
        "segment_sales": segment_sales,
        "state_sales": state_sales,
        "payment_method_sales": payment_sales,
    }

    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        frame.to_sql(name, engine, if_exists="replace", index=False, chunksize=10000, method="multi")

    print("Creating visualizations.")
    create_visualizations(monthly_sales, segment_sales, state_sales, payment_sales)

    print("Training and evaluating model.")
    metrics = train_model(cleaned)

    summary = f"""
Phase 2 Batch Processing Pipeline Summary

Raw records ingested: {len(df):,}
Cleaned records processed: {len(cleaned):,}
Model records used: {metrics['model_rows']:,}
Training rows: {metrics['train_rows']:,}
Testing rows: {metrics['test_rows']:,}

Model evaluation:
MAE: {metrics['mae']:.2f}
RMSE: {metrics['rmse']:.2f}
R2 Score: {metrics['r2']:.4f}

Generated outputs:
- cleaned_transactions.csv
- monthly_sales.csv
- quarterly_sales.csv
- segment_sales.csv
- state_sales.csv
- payment_method_sales.csv
- model_predictions.csv
- sales and model evaluation PNG graphs
- trained RandomForest model file

Reflection:
The implementation demonstrates a reproducible local batch-processing architecture using Docker containers.
The pipeline ingests timestamped e-commerce transactions, stores raw data in PostgreSQL, performs
preprocessing and aggregation, generates analytical visualizations, and trains a machine-learning model.
"""

    (OUTPUT_DIR / "phase2_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print("Processing completed successfully.")

if __name__ == "__main__":
    main()
