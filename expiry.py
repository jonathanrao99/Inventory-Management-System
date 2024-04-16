import pandas as pd

# Read data from CSV file
df = pd.read_csv("/Users/moraarvindkumar/Desktop/Major_project/data_set/data.csv")

# Convert Expiration_Date to datetime
df['Expiration_Date'] = pd.to_datetime(df['expiry_date'])

# Calculate days until expiration
df['Days_Until_Expiry'] = (df['Expiration_Date'] - pd.Timestamp.today()).dt.days

# Identify products nearing expiration (e.g., within 7 days)
near_expiry_threshold = 7
near_expiry_products = df[df['Days_Until_Expiry'] <= near_expiry_threshold]

# Recommend special offers or discounts for near-expiry products
for index, row in near_expiry_products.iterrows():
    print(f"Product {row['product_name']} is nearing expiration. Consider offering a special discount!")

# Display products nearing expiration
print("\nProducts nearing expiration within", near_expiry_threshold, "days:")
print(near_expiry_products)
