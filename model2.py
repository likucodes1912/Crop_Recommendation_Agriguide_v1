# ====================
# Data Preparation
# ====================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
# data = pd.read_excel('C:/Users/likul/Desktop/Project/AgriGuide/sampledata.xlsx')

data = pd.read_csv('C:/Users/likul/Desktop/Project/AgriGuide/crop_recommendation.csv')
# data = pd.read_csv('C:/Users/likul/Desktop/Project/AgriGuide/Soil_dataset.csv')

# Encode labels
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# Split features and target
X = data.drop('label', axis=1)
y = data['label']

# Split dataset (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================
# Machine Learning Model
# ====================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# Evaluate Random Forest
y_pred = rf_model.predict(X_test_scaled)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# print(classification_report(y_test, y_pred, target_names=le.classes_))

# ====================
# Deep Learning Model
# ====================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Calculate class weights
class_weights = {
    i: len(y) / (len(np.unique(y)) * np.bincount(y)[i])
    for i in np.unique(y)
}

# Build deep learning model
model = Sequential([
    Input(shape=(7,)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)
# history = model.fit(
#     X_train_scaled, y_train,
#     epochs=200,
#     validation_split=0.2,
#     class_weight=class_weights,
#     callbacks=[early_stop],
#     verbose=0
# )

# Evaluate deep learning model
dl_loss, dl_acc = model.evaluate(X_test_scaled, y_test)
print(f"\nNeural Network Accuracy: {dl_acc:.2f}")

# ====================
# Visualization
# ====================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Feature importance plot
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ====================
# Recommendation System
# ====================
def crop_recommender(N, P, K, temp, humidity, ph, rainfall):
    """
    Returns crop probabilities in descending order
    """
    inputs = scaler.transform([[N, P, K, temp, humidity, ph, rainfall]])

    # Get predictions from both models
    rf_proba = rf_model.predict_proba(inputs)[0]
    dl_proba = model.predict(inputs, verbose=0)[0]

    # Average probabilities
    avg_proba = (rf_proba + dl_proba) / 2

    # Create sorted dictionary
    crops = le.inverse_transform(np.arange(len(avg_proba)))
    return {
        crop: f"{prob:.1%}" for crop, prob in
        sorted(zip(crops, avg_proba), key=lambda x: x[1], reverse=True)
    }

# Example usage
print("\nCrop Recommendation:")
print(crop_recommender(90, 42, 43, 21, 82, 6.5, 203))
