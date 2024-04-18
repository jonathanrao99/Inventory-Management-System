import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings
import pickle

# Replace "data.csv" with the actual path to your data filejn
data = pd.read_csv("Inventory-Management-System-main/data_set/data.csv")

try:
    # Convert date column to datetime format (handle potential errors)
    data["date"] = pd.to_datetime(data["date_sale"], format='%d-%m-%Y')  # Specify the correct date format
except pd.errors.ParserError:
    print("Error: Unable to parse date column. Please check the data format.")
    exit()

# Try calculating basic sales metrics (handle potential missing columns)
try:
    total_sales = data["total_revenue"].sum()
    average_order_value = data["total_revenue"].mean()
except KeyError:
    print("Error: Columns 'total_revenue' not found in data. Please check column names.")
    exit()

# Top 5 selling products (by quantity sold)
try:
    top_selling_products = data.nlargest(5, "quantity_sold")  # Adjust column name if needed
except KeyError:
    print("Error: Column 'quantity' not found in data. Please check column names.")
    top_selling_products = None  # Set to None to avoid displaying empty DataFrame

# ------------------------ Basic Sales Analytics (if applicable) ------------------------

if not any([total_sales, average_order_value, top_selling_products]):  # Check if any metrics were calculated
    print("Warning: Unable to calculate basic sales metrics due to missing data.")
else:
    print("\nTotal Sales: ${:.2f}".format(total_sales))
    print("Average Order Value: ${:.2f}".format(average_order_value))
    print("\nTop 5 Selling Products:")
    print(top_selling_products) if top_selling_products is not None else print("Top 5 selling products data unavailable.")

# ------------------------ Time Series Analysis and Visualization ------------------------

# Resample data by month (or other desired frequency) and sum total revenue
try:
    resampled_data = data.resample("ME", on="date")["total_revenue"].sum()
except KeyError:
    print("Error: Column 'total_revenue' not found in data. Please check column names.")
    exit()

# Check if enough data points are available for seasonal decomposition
if resampled_data.shape[0] < 24:
    print("Warning: Not enough data points (less than 24) for reliable seasonal decomposition.")
else:
    # Perform seasonal decomposition (if data allows)
    decomposition = seasonal_decompose(resampled_data, model="additive")
    decomposition.plot()
    plt.show()

# Potential forecasting (using a simple example)
if resampled_data.shape[0] >= 12:  # Use at least 12 data points for forecasting
    try:
        # Define and fit an ARIMA model (adjust parameters as needed)
        model = ARIMA(resampled_data, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast for the next 3 months (adjust forecast horizon as needed)
        forecast = model_fit.forecast(steps=3)

        # Print the forecast results
        print("Forecast for the next 3 months:")
        print(forecast)

        # Visualize the forecast (optional)
        plt.figure(figsize=(12, 6))
        plt.plot(resampled_data.index, resampled_data.values, marker="o", linestyle="-", label="Actual")
        plt.plot(resampled_data.index[-3:], forecast, marker="o", linestyle="--", label="Forecast")
        plt.xlabel("Date")
        plt.ylabel("Total Revenue")
        plt.title("Sales Trend with Forecast")
        plt.grid(True)
        plt.legend()
        plt.show()
    except ValueError as e:  # Handle potential errors during model fitting
        print("Error during forecasting:", e)
else:
    print("Warning: Not enough data points (less than 12) for reliable forecasting.")

print("---------------------------------------------------------------------------")
