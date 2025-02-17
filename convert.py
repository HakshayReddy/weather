import pandas as pd

# Sample data (you can replace this with your actual dataset)
data = ['2019-01-01T00:00', '2019-01-01T06:00', '2019-01-01T12:00', '2019-01-01T18:00']

df = pd.read_csv("data.csv")

# Convert to datetime format
df['datetime'] = pd.to_datetime(df['time'])

# Extract the date part and time part in 24-hour format
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.strftime('%H:%M')
specific_times = ['00:00', '06:00', '12:00', '18:00']
df_filtered = df[df['time'].isin(specific_times)]
df_filtered.to_csv('data1.csv', index=False)