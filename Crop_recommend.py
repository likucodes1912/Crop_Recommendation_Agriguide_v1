import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("crop_recommendation.csv")

# Display first few rows of the dataset
print(df.head())

# Define features and target variable
X = df.drop(columns=["label"])
y = df["label"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict crop
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(data)
    return prediction[0]

# Example usage
N = 90
P = 42
K = 43
temperature = 25.3
humidity = 80.2
ph = 6.5
rainfall = 200.0

recommended_crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
print(f"Recommended Crop: {recommended_crop}")
