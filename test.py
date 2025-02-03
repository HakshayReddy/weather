import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. Load and Preprocess Data
df = pd.read_csv("GlobalWeatherRepository.csv")
df['last_updated'] = pd.to_datetime(df['last_updated'])
df.set_index('last_updated', inplace=True)

# Selecting important features
features = ['temperature_celsius', 'humidity', 'wind_kph', 'precip_mm']
df = df[features]

# Handling missing values
df.fillna(method='ffill', inplace=True)

# Lag Features for time dependencies
for lag in range(1, 4):
    for col in features:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

# Dropping NaN after shifting
df.dropna(inplace=True)

# Scaling	
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 2. Preparing Data for LSTM
# Sequence generator
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps, 0])  # Predicting temperature_celsius
    return np.array(X), np.array(y)

# Sequence length (look-back window)
time_steps = 10
X, y = create_sequences(df_scaled, time_steps)

# Train-test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. Building the LSTM Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer for temperature prediction

model.compile(optimizer='adam', loss='mse')

# 4. Training the Model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 5. Plotting Training History
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# 6. Making Predictions
y_pred = model.predict(X_test)

# Inverse scaling for predictions
y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((y_pred.shape[0], df.shape[1] - 1))), axis=1))[:, 0]
y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], df.shape[1] - 1))), axis=1))[:, 0]

# 7. Plotting Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Temperature')
plt.plot(y_pred_inv, label='Predicted Temperature')
plt.xlabel('Time Steps')
plt.ylabel('Temperature (Celsius)')
plt.title('Actual vs Predicted Temperature')
plt.legend()
plt.show()
