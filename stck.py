import numpy as np
import pandas as pd 
import yfinance as yf
import keras
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os

# Define and compile your model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))  # Adjust input_shape based on your data
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Save the model as an H5 file
model.save(r'C:\Users\LENOVO\OneDrive\Desktop\New folder\New folder\model.h5')
model.save(r'C:\Users\LENOVO\OneDrive\Desktop\New folder\New folder\model.keras')

model_path = r'C:\Users\LENOVO\OneDrive\Desktop\New folder\New folder\model.h5'  # or 'model.keras'

if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully.")
else:
    print(f"File not found: {model_path}")

st.header("Stock Market Predictor")
stock = st.text_input("Enter Stock Symbol", "GOOG")

# Fetch the most recent data
data = yf.download(stock, period='1y', interval='1d')  # Adjust the period and interval as needed

st.subheader('Stock Data')
st.write(data)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Prepare for future predictions
future_predictions = []
last_100_days = scaled_data[-100:]

for _ in range(30):  # Predict for the next 30 days
    last_100_days = last_100_days.reshape((1, last_100_days.shape[0], 1))
    next_pred = model.predict(last_100_days)
    future_predictions.append(next_pred[0, 0])
    last_100_days = np.append(last_100_days[0, 1:], next_pred).reshape(-1, 1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate future dates
future_dates = pd.date_range(start=data.index[-1], periods=31, freq='D')[1:]  # Skip the first date to get the next 30 days

# Plot original price with increased figure size
st.subheader('Original Stock Price')
fig1, ax1 = plt.subplots(figsize=(12, 8))  # Increased figure size
ax1.plot(data.index, data['Close'], label='Original Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

# Display original prices with time
original_df = pd.DataFrame({
    'Date': data.index,
    'Original Price': data['Close']
})
st.write(original_df)

# Plot future predictions with increased figure size
st.subheader('Future Stock Price Predictions')
fig2, ax2 = plt.subplots(figsize=(12, 8))  # Increased figure size
ax2.plot(future_dates, future_predictions, label='Future Predicted Price', linestyle='dotted',color='r')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Display future predictions day by day
future_df = pd.DataFrame({
    'Date': future_dates,
    'Future Predicted Price': future_predictions.flatten()
})
st.write(future_df)
