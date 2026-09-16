"""
Build data/retail.db from data/sample.csv.

Uses the cleaning logic already written by Member 1 (src/data/loader.py)
and the exact table/column mapping documented in reports/data_dictionary.md.

Usage (from the repo root):
    python scripts/build_database.py
    python scripts/build_database.py --csv data/sample.csv --out data/retail.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Make the repo root importable regardless of the current working directory
# or how this script is invoked (this is what "ModuleNotFoundError: No
# module named 'src'" means — Python only looks in the script's own folder
# unless we add the repo root ourselves).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import clean_data, validate_data


def build_database(csv_path: str, schema_path: str, db_path: str) -> None:
    csv_path, schema_path, db_path = Path(csv_path), Path(schema_path), Path(db_path)

    print(f"Reading {csv_path} ...")
    raw = pd.read_csv(csv_path)

    print("Cleaning with src.data.loader.clean_data() ...")
    df = clean_data(raw)
    validate_data(df)
    print(f"Cleaned rows: {len(df)}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # always rebuild from scratch, no stale data

    conn = sqlite3.connect(db_path)
    conn.executescript(schema_path.read_text(encoding="utf-8"))

    # ---- customers: one row per unique Customer ID ------------------------
    customers = (
        df[["Customer ID", "Country"]]
        .drop_duplicates(subset="Customer ID")
        .rename(columns={"Customer ID": "customer_id", "Country": "country"})
    )
    customers.to_sql("customers", conn, if_exists="append", index=False)

    # ---- products: one row per unique StockCode ----------------------------
    products = (
        df[["StockCode", "Description"]]
        .drop_duplicates(subset="StockCode")
        .rename(columns={"StockCode": "stock_code", "Description": "description"})
    )
    products.to_sql("products", conn, if_exists="append", index=False)

    # ---- orders: one row per unique Invoice --------------------------------
    orders = (
        df[["Invoice", "Customer ID", "InvoiceDate", "IsCancelled"]]
        .drop_duplicates(subset="Invoice")
        .rename(
            columns={
                "Invoice": "invoice_no",
                "Customer ID": "customer_id",
                "InvoiceDate": "invoice_date",
                "IsCancelled": "is_cancelled",
            }
        )
    )
    orders["is_cancelled"] = orders["is_cancelled"].astype(int)
    orders.to_sql("orders", conn, if_exists="append", index=False)

    # ---- order_items: one row per transaction line -------------------------
    order_items = df[["Invoice", "StockCode", "Quantity", "Price", "Revenue"]].rename(
        columns={
            "Invoice": "invoice_no",
            "StockCode": "stock_code",
            "Quantity": "quantity",
            "Price": "unit_price",
            "Revenue": "revenue",
        }
    )
    order_items.to_sql("order_items", conn, if_exists="append", index=False)

    conn.commit()

    # ---- sanity check: no orphan order_items --------------------------------
    orphans = conn.execute(
        "SELECT COUNT(*) FROM order_items oi "
        "LEFT JOIN orders o ON o.invoice_no = oi.invoice_no "
        "WHERE o.invoice_no IS NULL"
    ).fetchone()[0]
    conn.close()

    print(f"customers: {len(customers)} | products: {len(products)} | "
          f"orders: {len(orders)} | order_items: {len(order_items)}")
    print(f"orphan order_items: {orphans} (should be 0)")
    print(f"Database written to {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/sample.csv")
    parser.add_argument("--schema", default="sql/schema.sql")
    parser.add_argument("--out", default="data/retail.db")
    args = parser.parse_args()
    build_database(args.csv, args.schema, args.out)
