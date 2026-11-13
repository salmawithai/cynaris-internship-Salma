
import pandas as pd
import os

# ==========================================
# 1. LOAD DATASETS
# ==========================================

customers = pd.read_csv("data/customers.csv")
orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")

print("\n========== DATASET INFORMATION ==========")

# Display shape, data types and first 10 rows
for name, df in [
    ("CUSTOMERS", customers),
    ("ORDERS", orders),
    ("PRODUCTS", products)
]:
    print(f"\n{name}")
    print("Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 10 Records:")
    print(df.head(10))

# ==========================================
# 2. DATA INSPECTION
# ==========================================

print("\n========== DATA QUALITY CHECK ==========")

for name, df in [
    ("CUSTOMERS", customers),
    ("ORDERS", orders),
    ("PRODUCTS", products)
]:
    print(f"\n{name}")
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())

# ==========================================
# 3. FILTERING
# ==========================================

print("\n========== FILTERING ==========")

# Select orders where quantity is greater than 1
filtered_orders = orders[orders["Quantity"] > 1]

print("Orders with quantity greater than 1:")
print(filtered_orders.head(10))

# ==========================================
# 4. GROUPBY
# ==========================================

print("\n========== GROUPBY ==========")

# Calculate total quantity ordered for each product
product_summary = orders.groupby("Product_ID")["Quantity"].sum()

print("\nTotal quantity by product:")
print(product_summary.head(10))

# ==========================================
# 5. MERGE
# ==========================================

print("\n========== MERGING DATASETS ==========")

# Merge orders with customer information
merged_data = pd.merge(
    orders,
    customers,
    on="Customer_ID",
    how="left",
    validate="many_to_one"
)

# Merge the result with product information
merged_data = pd.merge(
    merged_data,
    products,
    on="Product_ID",
    how="left",
    suffixes=("_order", "_product"),
    validate="many_to_one"
)

print("Merged dataset shape:", merged_data.shape)
print(merged_data.head(10))

# ==========================================
# 6. CREATE REVENUE COLUMN
# ==========================================

# Calculate revenue using quantity multiplied by product price
merged_data["Revenue"] = (
    merged_data["Quantity"] * merged_data["Price (INR)"]
)

print("\nRevenue calculation:")
print(merged_data[["Order_ID", "Product_Name", "Quantity", "Revenue"]].head())

# ==========================================
# 7. PIVOT TABLE
# ==========================================

print("\n========== PIVOT TABLE ==========")

# Summarize revenue by customer city and product category
pivot = pd.pivot_table(
    merged_data,
    values="Revenue",
    index="City",
    columns="Category",
    aggfunc="sum",
    fill_value=0
)

print(pivot.head(10))

# ==========================================
# 8. EXPORT CLEANED DATA
# ==========================================

os.makedirs("data/processed", exist_ok=True)

# Convert date columns to datetime
merged_data["Order_Date"] = pd.to_datetime(
    merged_data["Order_Date"], errors="coerce"
)

merged_data["Delivery_Date"] = pd.to_datetime(
    merged_data["Delivery_Date"], errors="coerce"
)

# Export to CSV
csv_path = "data/processed/cleaned_sales.csv"
merged_data.to_csv(csv_path, index=False)

# Export to Parquet
parquet_path = "data/processed/cleaned_sales.parquet"
merged_data.to_parquet(parquet_path, index=False)

# ==========================================
# 9. COMPARE FILE SIZES
# ==========================================

csv_size = os.path.getsize(csv_path)
parquet_size = os.path.getsize(parquet_path)

print("\n========== FILE SIZE COMPARISON ==========")

print(f"CSV Size: {csv_size:,} bytes")
print(f"Parquet Size: {parquet_size:,} bytes")

if csv_size > parquet_size:
    print("Parquet file is smaller.")
else:
    print("CSV file is smaller.")

print("\nDay 2 Pandas implementation completed successfully!")
