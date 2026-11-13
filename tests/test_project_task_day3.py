
import unittest
import pandas as pd

from src.project_task_day3 import (
    clean_customers,
    clean_orders,
    clean_products,
    verify_foreign_keys,
    validate_columns
)


class TestDay3Workflow(unittest.TestCase):

    def test_clean_customers(self):
        data = {
            "Customer_ID": [" C001 "],
            "Name": [" Salma "],
            "City": [" Ongole "],
            "Contact_Number": ["98765-43210"],
            "Email": [" TEST@GMAIL.COM "],
            "Gender": [" Female "],
            "Address": [" Ongole "]
        }

        result = clean_customers(pd.DataFrame(data))

        self.assertEqual(result.loc[0, "Email"], "test@gmail.com")
        self.assertEqual(result.loc[0, "Contact_Number"], "9876543210")
        self.assertEqual(result.loc[0, "Name"], "Salma")

    def test_clean_orders(self):
        data = {
            "Order_ID": [1],
            "Customer_ID": [" C001 "],
            "Product_ID": [101],
            "Quantity": [2],
            "Order_Date": ["2026-11-01"],
            "Order_Time": [" 10:00 AM "],
            "Delivery_Date": ["2026-11-02"],
            "Delivery_Time": [" 11:00 AM "],
            "Location": [" Ongole "],
            "Occasion": [" Birthday "]
        }

        result = clean_orders(pd.DataFrame(data))

        self.assertEqual(result.loc[0, "Customer_ID"], "C001")
        self.assertEqual(result.loc[0, "Quantity"], 2)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["Order_Date"]))

    def test_clean_products(self):
        data = {
            "Product_ID": [101],
            "Product_Name": [" Cake "],
            "Category": [" Dessert "],
            "Price (INR)": [500],
            "Occasion": [" Birthday "]
        }

        result = clean_products(pd.DataFrame(data))

        self.assertEqual(result.loc[0, "Product_Name"], "Cake")
        self.assertEqual(result.loc[0, "Category"], "Dessert")

    def test_verify_foreign_keys(self):
        orders = pd.DataFrame({
            "Customer_ID": ["C001", "C999"],
            "Product_ID": [101, 999]
        })

        customers = pd.DataFrame({
            "Customer_ID": ["C001"]
        })

        products = pd.DataFrame({
            "Product_ID": [101]
        })

        result = verify_foreign_keys(orders, customers, products)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Customer_ID"], "C001")

    def test_validate_columns(self):
        data = pd.DataFrame({
            "Customer_ID": ["C001"]
        })

        with self.assertRaises(ValueError):
            validate_columns(
                data,
                ["Customer_ID", "Name"],
                "Customers"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
