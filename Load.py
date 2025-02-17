from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load your trained model
model = load_model("weather_prediction_model.h5", compile=False)
model.compile(optimizer='adam', loss='mse')

# Load dataset (same dataset used for training)
data = pd.read_csv("data.csv")  # Replace with actual data file

# Ensure 'datetime' column is in proper date format
data['datetime'] = pd.to_datetime(data['datetime'], format='%d-%m-%Y', errors='coerce')
data.dropna(subset=['datetime'], inplace=True)  # Drop rows with invalid datetime

# Sort data by date (if not already sorted)
data = data.sort_values(by='datetime')

# Define features used for training
features = ["temperature_celsius", "wind_kph", "cloud", "humidity", "precip_mm"]

# Load the training data again to fit the scaler
train_data = pd.read_csv("data.csv")  # Ensure this is the same dataset used for training

# Fit the MinMaxScaler on the training data only (to ensure same scaling is used)
scaler = joblib.load('scale.pkl')
scaler.fit(train_data[features])

# Function to get past 1-year data
def get_past_1_year_data(df, given_date):
    """
    Extracts the past 1 year of data for the model based on the given date.
    """
    given_date = pd.to_datetime(given_date, format='%d-%m-%Y', errors='coerce')
    if pd.isna(given_date):
        raise ValueError("Invalid date format. Use 'DD-MM-YYYY'.")

    # Convert index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df['datetime'], format='%d-%m-%Y', errors='coerce'))
    
    # Define start date (1 year before the given date)
    start_date = given_date - pd.DateOffset(years=1)

    # Select past 1-year data for the defined features
    past_year_data = df.loc[(df.index >= start_date) & (df.index <= given_date), features]
    
    if past_year_data.empty:
        raise ValueError("No data available for the past 1-year period.")

    return past_year_data


# Input date for prediction
given_date = "1-06-2022"

# Fetch past 1-year data for model input
try:
    past_data = get_past_1_year_data(train_data, given_date)
    
    # Scale past data using the same scaler from training
    past_data_scaled = scaler.transform(past_data)  
    
    # Reshape for LSTM input
    X_test = np.array([past_data_scaled])  
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))  # Ensure correct shape

except ValueError as e:
    print(f"Error: {e}")
    exit()

# Predict using LSTM model
prediction_scaled = model.predict(X_test)

# **Inverse Transform to get Actual Values**
predicted_weather = scaler.inverse_transform(prediction_scaled)[0]  

# Display results
print(f"\nPredicted Weather for {given_date}:")
print(f"Temperature (°C): {predicted_weather[0]:.2f}")
print(f"Wind Speed (kph): {predicted_weather[1]:.2f}")
print(f"Cloud Cover (%): {predicted_weather[2]:.2f}")
print(f"Humidity (%): {predicted_weather[3]:.2f}")
print(f"Precipitation (mm): {predicted_weather[4]:.2f}")


# Plot past 1 year data for temperature
plt.figure(figsize=(10, 5))
plt.plot(past_data.index[-365:], past_data["temperature_celsius"][-365:], label="Past 1 Year Temperature", marker='o')

# Plot predicted temperature
plt.scatter([pd.to_datetime(given_date, format='%d-%m-%Y', errors='coerce')], [predicted_weather[0]], color="red", label="Predicted Temperature", marker="s")

plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title(f"Weather Prediction for {given_date}")
plt.legend()
plt.xticks(rotation=45)
plt.show()

