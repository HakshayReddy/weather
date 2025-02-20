from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your trained MLP model
mlp_model = load_model("mlp_weather_model.h5", compile=False)
mlp_model.compile(optimizer='adam', loss='mse')

# Load scaler (same scaler used during training)
scaler = joblib.load('scale1.pkl')

# Load dataset (same dataset used during training)
data = pd.read_csv("data.csv")

# Ensure 'datetime' column is in proper date format
data['datetime'] = pd.to_datetime(data['datetime'], format='%d-%m-%Y', errors='coerce')
data.dropna(subset=['datetime'], inplace=True)  # Drop rows with invalid datetime

# Sort data by date (if not already sorted)
data = data.sort_values(by='datetime')

# Define features used during training
features = ["temperature_celsius", "wind_kph", "cloud", "humidity", "precip_mm"]

# Function to get past 1-year data
def get_past_1_year_data(df, given_date):

    given_date = pd.to_datetime(given_date, format='%d-%m-%Y', errors='coerce')
    if pd.isna(given_date):
        raise ValueError("Invalid date format. Use 'DD-MM-YYYY'.")

    # Ensure datetime index for filtering
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df['datetime'], format='%d-%m-%Y', errors='coerce'))
    
    # Define start date (1 year before the given date)
    start_date = given_date - pd.DateOffset(years=1)

    # Retrieve past 1-year data for the given features
    past_year_data = df.loc[(df.index >= start_date) & (df.index <= given_date), features]

    if past_year_data.empty:
        raise ValueError("No data available for the past 1-year period.")

    return past_year_data

# Input date for prediction
given_date = "8-11-2021"


past_data = get_past_1_year_data(data, given_date)

# Ensure we take exactly 365 days of past data
past_data = past_data.iloc[-365:]  # Take only the last 365 days

# Scale past data using the same scaler from training
past_data_scaled = scaler.transform(past_data)

# Flatten for MLP input
X_test = past_data_scaled.flatten().reshape(1, -1)  # Shape should now be (1, 1825)

# Check shape before prediction
print("Shape of input data for MLP:", X_test.shape)  # Should print (1, 1825)

# Predict using MLP model
mlp_prediction_scaled = mlp_model.predict(X_test)

# Inverse transform to get actual values
predicted_weather = scaler.inverse_transform(mlp_prediction_scaled)[0]

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load actual weather data for the given date
actual_weather = data.loc[data['datetime'] == given_date, features].values

if actual_weather.size == 0:
    raise ValueError(f"No actual data available for {given_date} to compute accuracy.")

# Avoid division by zero for MAPE
actual_weather[actual_weather == 0] = 1e-5  

# Reshape predicted_weather to match actual_weather
predicted_weather = predicted_weather.reshape(1, -1)

#  Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(actual_weather, predicted_weather) * 100

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(actual_weather, predicted_weather)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(actual_weather, predicted_weather))


# Print results
print(f"\nPredicted Weather for {given_date}:")
print(f"Temperature (°C): {predicted_weather[0][0]:.2f}")
print(f"Wind Speed (kph): {predicted_weather[0][1]:.2f}")
print(f"Cloud Cover (%): {predicted_weather[0][2]:.2f}")
print(f"Humidity (%): {predicted_weather[0][3]:.2f}")
print(f"Precipitation (mm): {predicted_weather[0][4]:.2f}")

print("\n**Performance Metrics:**")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

