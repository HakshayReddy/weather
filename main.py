from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("data.csv")

print(df.head())

#Indexing the data with the date 
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M')
df.set_index('time', inplace=True)

# View the transformed dataframe columns
print(df.columns)

# Summary statistics for the numerical columns
print(df.describe())

# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.savefig('correlations.png', dpi=300)
plt.show()

# Visualizing distributions of numerical columns
numerical_cols = df.columns.values[1:]

# Plotting histograms for each numerical feature
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    
plt.tight_layout()
plt.savefig(f'Distribution of Columns.png',dpi = 300)
plt.show()

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Plotting histograms for each numerical feature after 
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    
plt.tight_layout()
plt.savefig(f'Distribution of Columns After.png',dpi = 300)
plt.show()

print(df.describe())

# Create sequences for LSTM input
def create_sequences(data,seq_length):
    X,y = [],[]
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length]) # Sequence of past values
        y.append(data[i+seq_length]) # Corresponding future value (all columns)
    return np.array(X),np.array(y)

# Set sequence length
seq_length = 10

# Create sequences for input (X) and target (y)
X, y = create_sequences(df.values, seq_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshaping data for LSTM (samples, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

