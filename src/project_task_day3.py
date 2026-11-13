
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
project_task_day3.py
--------------------
Week 1 Day 3: Data Loading, Cleaning, Inspection and NumPy Summary

Author: Shaik Salma
Internship: AI/ML - 3 Months
"""

from pathlib import Path
import logging

import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# --------------------------------------------------
# 2. File Paths
# --------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed" / "day3"

CUSTOMERS_FILE = DATA_DIR / "customers.csv"
ORDERS_FILE = DATA_DIR / "orders.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"


# --------------------------------------------------
# 3. Load Datasets
# --------------------------------------------------

def load_datasets():

    try:
        customers = pd.read_csv(
            CUSTOMERS_FILE,
            dtype={
                "Customer_ID": "string",
                "Contact_Number": "string"
            }
        )

        orders = pd.read_csv(
            ORDERS_FILE,
            dtype={"Customer_ID": "string"}
        )

        products = pd.read_csv(PRODUCTS_FILE)

        logging.info("All datasets loaded successfully.")

        return customers, orders, products

    except FileNotFoundError as error:
        logging.error("Dataset file not found: %s", error)
        raise

    except pd.errors.ParserError as error:
        logging.error("CSV parsing error: %s", error)
        raise


# --------------------------------------------------
# 4. Validate Required Columns
# --------------------------------------------------

def validate_columns(df, required_columns, dataset_name):

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing_columns}"
        )


# --------------------------------------------------
# 5. Clean Customer Data
# --------------------------------------------------

def clean_customers(customers):

    required_columns = [
        "Customer_ID", "Name", "City",
        "Contact_Number", "Email", "Gender", "Address"
    ]

    validate_columns(
        customers, required_columns, "Customers"
    )

    for column in required_columns:
        customers[column] = (
            customers[column].astype("string").str.strip()
        )

    # Standardize email addresses
    customers["Email"] = customers["Email"].str.lower()

    # Keep phone numbers as strings and remove formatting characters
    customers["Contact_Number"] = (
        customers["Contact_Number"]
        .str.replace(r"[^\d]", "", regex=True)
    )

    # Remove exact duplicate rows
    customers = customers.drop_duplicates()

    return customers


# --------------------------------------------------
# 6. Clean Order Data
# --------------------------------------------------

def clean_orders(orders):

    required_columns = [
        "Order_ID", "Customer_ID", "Product_ID",
        "Quantity", "Order_Date", "Order_Time",
        "Delivery_Date", "Delivery_Time",
        "Location", "Occasion"
    ]

    validate_columns(orders, required_columns, "Orders")

    text_columns = [
        "Customer_ID", "Order_Time",
        "Delivery_Time", "Location", "Occasion"
    ]

    for column in text_columns:
        orders[column] = (
            orders[column].astype("string").str.strip()
        )

    # Convert date columns
    for column in ["Order_Date", "Delivery_Date"]:

        orders[column] = pd.to_datetime(
            orders[column],
            format="mixed",
            errors="coerce"
        )

    # Convert Quantity to numeric
    orders["Quantity"] = pd.to_numeric(
        orders["Quantity"],
        errors="coerce"
    )

    # Check for invalid or non-positive quantities
    invalid_quantity = (
        orders["Quantity"].isna() |
        (orders["Quantity"] <= 0) |
        (orders["Quantity"] % 1 != 0)
    )

    if invalid_quantity.any():
        logging.warning(
            "Found %d invalid quantity rows.",
            invalid_quantity.sum()
        )

    # Remove exact duplicate rows
    orders = orders.drop_duplicates()

    return orders


# --------------------------------------------------
# 7. Clean Product Data
# --------------------------------------------------

def clean_products(products):

    required_columns = [
        "Product_ID", "Product_Name",
        "Category", "Price (INR)", "Occasion"
    ]

    validate_columns(products, required_columns, "Products")

    text_columns = [
        "Product_Name", "Category", "Occasion"
    ]

    for column in text_columns:
        products[column] = (
            products[column].astype("string").str.strip()
        )

    # Remove exact duplicate rows
    products = products.drop_duplicates()

    return products


# --------------------------------------------------
# 8. Verify Foreign Keys
# --------------------------------------------------

def verify_foreign_keys(orders, customers, products):

    valid_customers = orders["Customer_ID"].isin(
        customers["Customer_ID"]
    )

    valid_products = orders["Product_ID"].isin(
        products["Product_ID"]
    )

    invalid_rows = ~(valid_customers & valid_products)

    logging.info(
        "Orders with invalid customer/product references: %d",
        invalid_rows.sum()
    )

    orders = orders[~invalid_rows].copy()

    return orders.reset_index(drop=True)


# --------------------------------------------------
# 9. Inspect Datasets
# --------------------------------------------------

def inspect_dataset(name, df):

    print(f"\n========== {name.upper()} ==========")

    print("Shape:", df.shape)

    print("Missing Values:")
    print(df.isnull().sum())

    print("Duplicate Rows:", df.duplicated().sum())

    print("Data Types:")
    print(df.dtypes)

    print("-" * 50)


# --------------------------------------------------
# 10. Save Cleaned Datasets
# --------------------------------------------------

def save_datasets(customers, orders, products):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customers.to_csv(
        OUTPUT_DIR / "cleaned_customers.csv",
        index=False
    )

    orders.to_csv(
        OUTPUT_DIR / "cleaned_orders.csv",
        index=False
    )

    products.to_csv(
        OUTPUT_DIR / "cleaned_products.csv",
        index=False
    )

    logging.info("All cleaned datasets saved successfully.")


# --------------------------------------------------
# 11. NumPy Quantity Summary
# --------------------------------------------------

def numpy_summary(orders):

    quantities = orders["Quantity"].to_numpy(
        dtype=np.float64,
        na_value=np.nan
    )

    valid_quantities = quantities[~np.isnan(quantities)]

    print("\n========== NUMPY SUMMARY ==========")

    if valid_quantities.size == 0:
        print("No valid quantity values available.")
        return

    print("Total Quantity:", np.sum(valid_quantities))

    print(
        "Average Quantity:",
        round(np.mean(valid_quantities), 2)
    )

    print("Minimum Quantity:", np.min(valid_quantities))

    print("Maximum Quantity:", np.max(valid_quantities))


# --------------------------------------------------
# 12. Main Execution
# --------------------------------------------------

def main():

    try:

        # Load datasets
        customers, orders, products = load_datasets()

        # Clean datasets
        customers = clean_customers(customers)
        orders = clean_orders(orders)
        products = clean_products(products)

        # Verify relationships
        orders = verify_foreign_keys(
            orders, customers, products
        )

        # Inspect datasets
        inspect_dataset("Customers", customers)
        inspect_dataset("Orders", orders)
        inspect_dataset("Products", products)

        # Save cleaned datasets
        save_datasets(customers, orders, products)

        # NumPy calculations
        numpy_summary(orders)

        logging.info("Day 3 task completed successfully.")

    except (ValueError, FileNotFoundError, pd.errors.ParserError) as error:
        logging.error("Day 3 task failed: %s", error)
        raise


if __name__ == "__main__":
    main()



