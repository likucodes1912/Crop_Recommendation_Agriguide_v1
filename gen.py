import pandas as pd
import random

# Load dataset
df = pd.read_csv('Soil_dataset.csv')

# Fill 'weather_condition' based on temperature, humidity, and rainfall
def assign_weather_condition(row):
    if row['rainfall'] > 200:
        return 'Rainy'
    elif row['rainfall'] < 50 and row['humidity'] < 40:
        return 'Dry'
    elif row['temperature'] > 30:
        return 'Hot'
    elif row['temperature'] < 20:
        return 'Cool'
    else:
        return 'Clear'

df['weather_condition'] = df.apply(lambda row: assign_weather_condition(row), axis=1)

# Fill 'wind_speed' based on weather condition
def assign_wind_speed(condition):
    if condition == 'Rainy':
        return round(random.uniform(4, 8), 1)
    elif condition == 'Clear':
        return round(random.uniform(3, 6), 1)
    elif condition == 'Hot':
        return round(random.uniform(2, 5), 1)
    else:  # Default for other conditions
        return round(random.uniform(5, 9), 1)

df['wind_speed'] = df['weather_condition'].apply(assign_wind_speed)

# Fill 'weather_suitability' based on crop and weather condition
def assign_weather_suitability(row):
    if row['label'] == 'rice' and row['weather_condition'] in ['Rainy', 'Humid']:
        return 'High'
    elif row['label'] == 'maize' and row['weather_condition'] in ['Sunny', 'Clear']:
        return 'High'
    else:
        return 'Medium'

df['weather_suitability'] = df.apply(lambda row: assign_weather_suitability(row), axis=1)

# Save the updated dataset
df.to_csv('Updated_Soil_dataset.csv', index=False)
