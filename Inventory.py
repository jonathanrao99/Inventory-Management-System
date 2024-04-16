import pandas as pd

def main():
    # Read inventory data from CSV file
    inventory_df = pd.read_csv("/Users/moraarvindkumar/Desktop/Major_project/data_set/data.csv")

    # Display current stock levels
    print("\n**Current Stock Levels:**")
    print(inventory_df[["product_id", "product_name", "quantity_stock"]].to_string())

    # Check for low stock and suggest restocking
    for index, row in inventory_df.iterrows():
        if row["quantity_stock"] <= 10:
            stock_difference = 10 - row["quantity_stock"]
            print(f"\n**Low Stock Alert:** {row['product_name']} (ID: {row['product_id']})")
            print(f"Current Quantity: {row['quantity_stock']}")
            print(f"Suggesting restock of {stock_difference} units to reach minimum stock level.")

if __name__ == "__main__":
    main()