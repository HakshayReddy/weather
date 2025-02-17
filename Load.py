from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your trained model
model = load_model("weather_prediction_model.h5", compile=False)
model.compile(optimizer='adam', loss='mse')

# Load dataset
data = pd.read_csv("data.csv")  # Replace with actual data file

# Ensure 'datetime' column is in proper date format
data['datetime'] = pd.to_datetime(data['datetime'], format='%d-%m-%Y %H:%M', errors='coerce')

# Sort data by date (if not already sorted)
data = data.sort_values(by='datetime')

# Define features used for training
features = ["temperature_celsius", "humidity", "feels_like_celsius", "precip_mm", 
            "pressure_mb", "cloud", "wind_kph", "wind_degree"]

# Function to get past 1-year data
def get_past_1_year_data(df, given_date):
    """
    Extracts the past 1 year of data for the model based on the given date.
    """
    given_date = pd.to_datetime(given_date, format='%d-%m-%Y %H:%M', errors='coerce')
    if pd.isna(given_date):
        raise ValueError("Invalid date format. Use 'DD-MM-YYYY HH:MM'.")

    # Convert index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M', errors='coerce'))
    
    # Define start date (1 year before the given date)
    start_date = given_date - pd.DateOffset(years=1)

    # Select past 1-year data for the defined features
    past_year_data = df.loc[(df.index >= start_date) & (df.index <= given_date), features]
    
    if past_year_data.empty:
        raise ValueError("No data available for the past 1-year period.")

    return past_year_data


# Input date for prediction
given_date = "15-01-2025 00:00"

# Fetch past 1-year data for model input
try:
    past_data = get_past_1_year_data(data, given_date)
    X_test = np.array([past_data.values])  # Get the entire 1 year data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))  # Ensure correct shape
except ValueError as e:
    print(f"Error: {e}")
    exit()

# Predict using LSTM model
prediction = model.predict(X_test)

# Debug: Check raw model output
print("Raw model output (before inverse transform):", prediction)

# Fix inverse transform to avoid extreme values
predicted_weather_full = np.zeros((1, len(features)))
predicted_weather_full[:, :prediction.shape[1]] = prediction
predicted_weather = predicted_weather_full[0]

# Display results
print(f"\nPredicted Weather for {given_date}:")
print(f"Temperature (°C): {predicted_weather[0]:.2f}")
print(f"Humidity (%): {predicted_weather[1]:.2f}")
print(f"Feels Like (°C): {predicted_weather[2]:.2f}")
print(f"Precipitation (mm): {predicted_weather[3]:.2f}")
print(f"Pressure (mb): {predicted_weather[4]:.2f}")
print(f"Cloud Cover (%): {predicted_weather[5]:.2f}")
print(f"Wind Speed (kph): {predicted_weather[6]:.2f}")
print(f"Wind Degree: {predicted_weather[7]:.2f}")

# Plot past 1 year data for temperature
plt.figure(figsize=(10, 5))
plt.plot(past_data.index[-365:], past_data["temperature_celsius"][-365:], label="Past 1 Year Temperature", marker='o')

# Plot predicted temperature
plt.scatter([pd.to_datetime(given_date)], [predicted_weather[0]], color="red", label="Predicted Temperature", marker="s")

plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title(f"Weather Prediction for {given_date}")
plt.legend()
plt.xticks(rotation=45)
plt.show()
