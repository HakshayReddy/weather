from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Load the data
df = pd.read_csv("GlobalWeatherRepository.csv")
df = df.drop(["timezone","last_updated_epoch","air_quality_Carbon_Monoxide","air_quality_Ozone","air_quality_Nitrogen_dioxide","air_quality_Sulphur_dioxide","air_quality_PM2.5","air_quality_PM10","air_quality_us-epa-index","air_quality_gb-defra-index","sunrise","sunset","moonrise","moonset","moon_phase","moon_illumination",
              "gust_kph", "visibility_miles","feels_like_fahrenheit","precip_in","pressure_in","wind_mph","temperature_fahrenheit","latitude","longitude","wind_direction"],axis = 1)

#Indexing the data with the date 
df['last_updated'] = pd.to_datetime(df['last_updated'], format='%Y-%m-%d %H:%M')
df = df.rename(columns = {'last_updated' : 'date'})
df.set_index('date', inplace=True)


# Initialize a LabelEncoder for each column
le_country = LabelEncoder()
le_location_name = LabelEncoder()
le_condition_text = LabelEncoder()

# Apply Label Encoding to each categorical column
df['country_encoded'] = le_country.fit_transform(df['country'])
df['location_name_encoded'] = le_location_name.fit_transform(df['location_name'])
df['condition_text_encoded'] = le_condition_text.fit_transform(df['condition_text'])

# Drop the original categorical columns since they are no longer needed
df.drop(['country', 'location_name', 'condition_text'], axis=1, inplace=True)

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

# Dropping 'gust_mph' due to its high correlation with 'wind_kph'
df = df.drop(columns=['gust_mph'])

# Visualizing distributions of numerical columns
numerical_cols = ['temperature_celsius', 'wind_kph', 'wind_degree', 'pressure_mb', 
                  'precip_mm', 'humidity', 'cloud', 'feels_like_celsius', 
                  'visibility_km', 'uv_index']

# Plotting histograms for each numerical feature
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    
plt.tight_layout()
plt.savefig(f'Distribution of Columns.png',dpi = 300)
plt.show()

# Applying log transformation on skewed features
df['precip_mm'] = df['precip_mm'].apply(lambda x: np.log1p(x) if x > 0 else 0)
df['cloud'] = df['cloud'].apply(lambda x: np.log1p(x) if x > 0 else 0)

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
