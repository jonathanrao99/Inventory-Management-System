from flask import Flask, request, render_template
from flask import render_template, request, jsonify
import pandas as pd
import pickle
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming 'model' is your trained Keras model


app = Flask(__name__)

# Load the pickled model
model_file_path = "trained_model.pkl"
try:
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None  # Set model to None if loading fails
    
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the file to a desired location
    file_path = "/Users/moraarvindkumar/Desktop/Major_project/data_set/data.csv"
    file.save(file_path)
    
    return "File uploaded successfully!"

def recommend_near_expiry_products(data):
    # Convert Expiration_Date to datetime
    data['Expiration_Date'] = pd.to_datetime(data['expiry_date'])

    # Filter out products with past expiration dates
    data = data[data['Expiration_Date'] >= pd.Timestamp.today()]

    # Calculate days until expiration
    # Using .loc to set values without triggering SettingWithCopyWarning
    data.loc[:, 'Days_Until_Expiry'] = (data['Expiration_Date'] - pd.Timestamp.today()).dt.days



    # Identify products nearing expiration (e.g., within 7 days)
    near_expiry_threshold = 7
    near_expiry_products = data[data['Days_Until_Expiry'] <= near_expiry_threshold]

    # Remove duplicate products
    near_expiry_products = near_expiry_products.drop_duplicates(subset=['product_id'])

    # Convert DataFrame to list of dictionaries for easy rendering in HTML
    recommendations = near_expiry_products.to_dict(orient='records')

    return recommendations

@app.route('/inventory')
def inventory():
    # Read data from CSV file
    data_file_path = ("/Users/moraarvindkumar/Desktop/Major_project/data_set/data.csv")
    df = pd.read_csv(data_file_path)

    # Get recommendations for restocking and near expiry products
    low_stock_recommendations = df[df['quantity_stock'] <= 300][['product_id', 'product_name']]
    near_expiry_recommendations = recommend_near_expiry_products(df)

    return render_template('inventory.html', restock_recommendations=low_stock_recommendations.to_dict(orient='records'),
                           near_expiry_recommendations=near_expiry_recommendations)

@app.route('/predict', methods=["POST"])
def predict():
    try:
    # Load the dataset (assuming the dataset path is configurable)
        dataset_path = "/Users/moraarvindkumar/Desktop/Major_project/data_set/data.csv"
        df = pd.read_csv(dataset_path)

        # Prepare input data for prediction (example: using the last three terms of quantity_sold)
        last_three_terms = df['quantity_sold'].values[-3:].reshape(1, -1)

        # Make predictions using the loaded model
        if model:
            predictions = model.predict(last_three_terms)
            prediction_value = predictions[0]  # Assuming single prediction output
        else:
            prediction_value = "Model not available"

        # Return prediction result as JSON response (assuming API endpoint)
        return jsonify({
            "prediction": float(prediction_value)  # Convert prediction to a float value
        })

    except FileNotFoundError as e:
        # Handle file not found error when loading the dataset
        error_message = f"Error loading dataset: {str(e)}"
        return jsonify({"error": error_message}), 500  # Return JSON error response with status code 500

    except Exception as e:
        # Handle generic exceptions during prediction
        error_message = f"Error making prediction: {str(e)}"
        return jsonify({"error": error_message}), 500 

@app.route('/analytics')
def sales_analytics():
    # Load data from CSV file
    data = pd.read_csv("/Users/moraarvindkumar/Desktop/Major_project/data_set/data.csv")

    # Calculate total sales and average order value
    total_sales = data["total_revenue"].sum()
    average_order_value = data["total_revenue"].mean()

    # Find top 5 selling and bottom 5 selling products based on quantity sold
    top_selling_products = data.nlargest(5, "quantity_stock")
    bottom_selling_products = data.nsmallest(5, "quantity_stock")

    # Pass DataFrames to the template directly
    return render_template('analytics.html', total_sales=total_sales,
                           average_order_value=average_order_value,
                           top_selling_products=top_selling_products,
                           bottom_selling_products=bottom_selling_products)


    # Plot sales trend
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales.index, monthly_sales['total_revenue'], marker='o', linestyle='-')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a static file
    sales_trend_file_path = "static/sales_trend.png"
    plt.savefig(sales_trend_file_path)

    return render_template('sales_trend.html')

if __name__ == '__main__':
    app.run(debug=True)
    