from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Load the data
df = pd.read_csv("data.csv")

# Indexing the data with the date
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y')
df.set_index('datetime', inplace=True)

# Define numerical features
numerical_cols = ["temperature_celsius", "wind_kph", "cloud", "humidity", "precip_mm"]

# Scale the features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the scaler for future use
joblib.dump(scaler, 'scale1.pkl')

# Create sequences for MLP input
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].flatten())  # Flatten to 1D for MLP
        y.append(data[i+seq_length])  # Corresponding future value (all columns)
    return np.array(X), np.array(y)

# Set sequence length
seq_length = 365

# Create sequences for input (X) and target (y)
X, y = create_sequences(df.values, seq_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build MLP model
mlp_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(numerical_cols))  # Output layer for multiple weather features
])

# Compile the model
mlp_model.compile(optimizer='adam', loss='mse')

# Train the model
history=mlp_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save trained model
mlp_model.save("mlp_weather_model.h5")

# Evaluate the model
loss= mlp_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
