from flask import Flask, request, render_template
from flask import render_template, request, jsonify
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming 'model' is your trained Keras model


app = Flask(__name__)

# Load the pickled model
model_file_path = "Inventory-Management-System-main/trained_model.pkl"
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
    file_path = "Inventory-Management-System-main/data_set/data.csv"
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
    data_file_path = ("Inventory-Management-System-main/data_set/data.csv")
    df = pd.read_csv(data_file_path)

    # Get recommendations for restocking and near expiry products
    low_stock_recommendations = df[df['quantity_stock'] <= 300][['product_id', 'product_name']]
    near_expiry_recommendations = recommend_near_expiry_products(df)

    return render_template('inventory.html', restock_recommendations=low_stock_recommendations.to_dict(orient='records'),
                           near_expiry_recommendations=near_expiry_recommendations)

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Extract input data from the request
            quantity1 = float(request.json['quantity1'])
            quantity2 = float(request.json['quantity2'])
            quantity3 = float(request.json['quantity3'])

            # Prepare the input data for prediction
            input_data = np.array([[quantity1, quantity2, quantity3]])

            # Make predictions using the loaded model
            if model:
                prediction_value = model.predict(input_data)[0][0]  # Assuming single prediction output
            else:
                prediction_value = "Model not available"

            # Return prediction result as JSON response
            return jsonify({
                "prediction": float(prediction_value)  # Convert prediction to a float value
            })

        except Exception as e:
            # Handle any exceptions that occur during prediction
            error_message = f"Failed to make prediction: {str(e)}"
            return jsonify({"error": error_message}), 500

    elif request.method == "GET":
        # Render the prediction page template
        return render_template("prediction.html")


@app.route('/analytics')
def sales_analytics():
    # Load data from CSV file
    data = pd.read_csv("Inventory-Management-System-main/data_set/data.csv")

    # Calculate total sales and average order value
    total_sales = data["total_revenue"].sum()
    average_order_value = data["total_revenue"].mean()

    # Find top 5 selling and bottom 5 selling products based on quantity sold
    top_selling_products = data.nlargest(5, "quantity_stock")
    bottom_selling_products = data.nsmallest(5, "quantity_stock")

    # Convert DataFrames to dictionaries
    top_selling_dict = top_selling_products.to_dict(orient='records')
    bottom_selling_dict = bottom_selling_products.to_dict(orient='records')

    # Plot sales trend
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['date_sale'], data['total_revenue'], marker='o', linestyle='-')
    ax.set_title('Monthly Sales Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Sales')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    fig.tight_layout()

    # Save the plot to a static file
    sales_trend_file_path = "sales_trend.png"
    fig.savefig(sales_trend_file_path)

    # Pass DataFrames to the template directly
    return render_template('analytics.html', total_sales=total_sales,
                           average_order_value=average_order_value,
                           top_selling_products=top_selling_dict,
                           bottom_selling_products=bottom_selling_dict)


if __name__ == '__main__':
    app.run(debug=True)
    
