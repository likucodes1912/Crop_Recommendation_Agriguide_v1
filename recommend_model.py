import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('Soil_dataset.csv')

# Data Preprocessing
# Drop columns that won't be used as features (weather_condition, weather_suitability)
features = data.drop(['label', 'weather_condition', 'weather_suitability'], axis=1)
target = data['label']

# Encode categorical labels
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and label encoder for future use
joblib.dump(model, 'crop_recommendation_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')


# Function to make recommendations
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall, wind_speed):
    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall],
        'wind_speed': [wind_speed]
    })

    # Load the model and encoder
    model = joblib.load('crop_recommendation_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # Make prediction
    prediction = model.predict(input_data)
    recommended_crop = label_encoder.inverse_transform(prediction)[0]

    # Get probabilities for all crops
    probabilities = model.predict_proba(input_data)[0]
    crop_probabilities = {label_encoder.classes_[i]: prob for i, prob in enumerate(probabilities)}
    sorted_crops = sorted(crop_probabilities.items(), key=lambda x: x[1], reverse=True)

    return recommended_crop, sorted_crops


# Interactive interface
def interactive_recommendation():
    print("Crop Recommendation System")
    print("Please enter the following parameters:")

    N = float(input("Nitrogen (N) level in soil (kg/ha): "))
    P = float(input("Phosphorus (P) level in soil (kg/ha): "))
    K = float(input("Potassium (K) level in soil (kg/ha): "))
    temperature = float(input("Temperature in °C: "))
    humidity = float(input("Humidity in %: "))
    ph = float(input("Soil pH: "))
    rainfall = float(input("Rainfall in mm: "))
    wind_speed = float(input("Wind speed in km/h: "))

    recommended_crop, all_crops = recommend_crop(N, P, K, temperature, humidity, ph, rainfall, wind_speed)

    print("\nRecommendation Results:")
    print(f"Best crop to grow: {recommended_crop}")
    print("\nAll crop probabilities:")
    for crop, prob in all_crops:
        print(f"{crop}: {prob:.2%}")


# Uncomment to run the interactive interface
# interactive_recommendation()

# VALIDATION
# Soil pH values (3.8-7.8) fit the standard pH scale for agriculture
# Rice appears in high-rainfall (>200mm) entries as expected
# Crops like chickpea appear in dry conditions (low humidity, high wind speed)
if __name__ == "__main__":
    # Example values
    example_N = int(input('Enter the Nitrogen Value (kg/ha):')) #N= 90 (kilograms per hectare)
    example_P = int(input('Enter the Phospohrous Value (kg/ha):')) #P= 42  (kilograms per hectare)
    example_K = int(input('Enter the Pottasium Value (kg/ha):')) # K= 43 (kilograms per hectare)
    example_temp = float(input('Enter the Temperature Value (°C):')) #temp= 20.87
    example_humidity = float(input('Enter the Humidity Value (%):')) #humidity= 82.0 (relative humidity)
    example_ph = float(input('Enter the pH Value(0-14):')) #ph_value= 6.5   (0-14, unitless)
    example_rainfall = float(input('Enter the Rainfall Value (mm):')) #rainfall= 202.9 (millimeters)
    example_wind = float(input('Enter the Wind Speed Value (km/h):')) #wind= 5.2  (kilometers per hour)

    recommended, probabilities = recommend_crop(example_N, example_P, example_K, example_temp,example_humidity, example_ph, example_rainfall, example_wind)

    print(f"Recommended crop: {recommended}")
    print("All probabilities:")
    for crop, prob in probabilities:
        print(f"{crop}: {prob:.2%}")


# NOTES
# The soil nutrient values (N, P, K) are likely available nutrient content (not total composition), which is standard for crop recommendation systems.
# Temperature values (20-36°C) match typical agricultural ranges.
# Humidity values (15-94%) confirm the unit is percentage.
# Rainfall values (43-263mm) suggest measurement over a growing season or monthly accumulation (not daily).
# Wind speed values (3.2-11.2 km/h) are realistic for agricultural conditions.


