from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('/Users/moraarvindkumar/Desktop/Major_project/data_set/data.csv')

# Preprocess data
look_back = 3

# Convert 'date_sale' column to datetime format (assuming 'date_sale' is in format 'dd-mm-yyyy')
df['date_sale'] = pd.to_datetime(df['date_sale'], format='%d-%m-%Y')

# Set 'date_sale' column as the index (assuming 'date_sale' contains unique timestamps)
df.set_index('date_sale', inplace=True)

# Optionally, you can sort the DataFrame by the index (date_sale) for chronological order
df.sort_index(inplace=True)

# Create lagged features (shifted values) for time series analysis (assuming look_back is 3)
for i in range(1, look_back + 1):
    df[f'sale_lag_{i}'] = df['quantity_sold'].shift(i)

# Drop rows with NaN values resulting from lagged features creation (due to shifting)
df.dropna(inplace=True)
# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
df['quantity_sold'] = scaler.fit_transform(df[['quantity_sold']])

def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Prepare the data
X, y = create_dataset(df['quantity_sold'].values, look_back)

# Split the data into train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]  # Fix the unexpected indentation
# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2)
# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform([y_test])
# Plotting
plt.ioff()  # Turn off interactive mode
plt.plot(df.index[look_back:look_back + len(train_predict)], train_predict, label='Train Prediction')
plt.plot(df.index[look_back + len(train_predict):], test_predict.flatten(), label='Test Prediction')
plt.plot(df.index[look_back:], df['quantity_sold'][look_back:], label='Actual Data')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Quantity Sold', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.savefig('static/plot.png')  # Save the plot as an image file
plt.close()

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_inv.flatten(), test_predict.flatten())
# Mean Squared Error (MSE)
mse = mean_squared_error(y_test_inv.flatten(), test_predict.flatten())
# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
# R-squared (R2)
# Mean Absolute Error (MAE)

def main():
    r2 = r2_score(y_test_inv.flatten(), test_predict.flatten())

    # Save the trained model as a pickle file
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)


