# TRY 3
# ====================
# Data Preparation
# ====================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv('Soil_dataset.csv')

# Separate features and target
X = data.drop('label', axis=1)
y = data['label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Define preprocessing for numeric vs categorical columns
numeric_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'wind_speed']
categorical_features = ['weather_condition', 'weather_suitability']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split dataset (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42)

# Create preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the training data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)


# MACHINE LEARNING MODEL
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_processed, y_train)

# Evaluate Random Forest
y_pred = rf_model.predict(X_test_processed)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# DEEP LEARNING MODEL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Calculate class weights
class_weights = {
    i: len(y) / (len(np.unique(y)) * np.bincount(y_encoded)[i])
    for i in np.unique(y_encoded)
}

# Build deep learning model
input_shape = X_train_processed.shape[1]
model = Sequential([
    Input(shape=(input_shape,)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_processed, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate deep learning model
dl_loss, dl_acc = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"\nNeural Network Accuracy: {dl_acc:.2f}")

# ====================
# Visualization
# ====================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Feature importance plot (for Random Forest)
numeric_feature_names = numeric_features
categorical_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
    categorical_features)
all_feature_names = np.concatenate([numeric_feature_names, categorical_feature_names])

importances = rf_model.feature_importances_
sorted_idx = importances.argsort()

plt.figure(figsize=(12, 8))
sns.barplot(x=importances[sorted_idx], y=all_feature_names[sorted_idx])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ====================
# Recommendation System
# ====================
def crop_recommender(input_data):
    """
    Returns crop recommendations with probabilities in descending order

    Parameters:
    input_data (dict): Dictionary containing all required features:
        - N, P, K: Soil nutrients
        - temperature: in °C
        - humidity: in %
        - ph: soil pH
        - rainfall: in mm
        - weather_condition: ['Clear', 'Rainy', 'Cool', 'Hot']
        - weather_suitability: ['High', 'Medium']
        - wind_speed: in km/h
    """
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])

    # Preprocess the input
    processed_input = pipeline.transform(input_df)

    # Get predictions from both models
    rf_proba = rf_model.predict_proba(processed_input)[0]
    dl_proba = model.predict(processed_input, verbose=0)[0]

    # Average probabilities
    avg_proba = (rf_proba + dl_proba) / 2

    # Create sorted recommendations
    crops = le.inverse_transform(np.arange(len(avg_proba)))
    recommendations = sorted(zip(crops, avg_proba), key=lambda x: x[1], reverse=True)

    # Format output
    result = {
        'top_recommendation': recommendations[0][0],
        'all_recommendations': [
            {'crop': crop, 'probability': f"{prob:.1%}", 'suitability': 'High' if prob > 0.5 else 'Medium'}
            for crop, prob in recommendations
        ]
    }

    return result


# Example usage
sample_input = {
    'N': 104,
    'P': 18,
    'K': 30,
    'temperature': 23.6,
    'humidity': 60.3,
    'ph': 6.7 ,
    'rainfall': 140.93,
    'weather_condition': 'Clear',
    'weather_suitability': 'Medium',
    'wind_speed': 5.4
}
print(type(sample_input))

print("\nCrop Recommendation:")
recommendations = crop_recommender(sample_input)
print(f"Top Recommendation: {recommendations['top_recommendation']}")
print("\nAll Recommendations:")
for rec in recommendations['all_recommendations']:
    print(f"- {rec['crop']}: {rec['probability']} ({rec['suitability']})")


# try 1
# # # ====================
# # # Data Preparation
# # # ====================
# # import pandas as pd
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.model_selection import train_test_split
# #
# # # Load dataset
# # data = data = pd.read_excel('C:/Users/likul/Desktop/Project/AgriGuide/sampledata.xlsx')
# # # data = csv.reader(data)
# #
# # # Encode labels
# # le = LabelEncoder()
# # data['label'] = le.fit_transform(data['label'])
# #
# # # Split features and target
# # X = data.drop('label', axis=1)
# # y = data['label']
# #
# # # Split dataset (70-30 split)
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.3, random_state=42
# # )
# #
# # # Standardize features
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)
# #
# # # ====================
# # # Machine Learning Model
# # # ====================
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, classification_report
# #
# # # Initialize and train Random Forest
# # rf_model = RandomForestClassifier(
# #     n_estimators=200,
# #     max_depth=5,
# #     class_weight='balanced',
# #     random_state=42
# # )
# # rf_model.fit(X_train_scaled, y_train)
# #
# # # Evaluate
# # y_pred = rf_model.predict(X_test_scaled)
# # print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# # # print(classification_report(y_test, y_pred, target_names=le.classes_))
# #
# # # ====================
# # # Deep Learning Model
# # # ====================
# # import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Dropout
# # from tensorflow.keras.callbacks import EarlyStopping
# # from tensorflow.keras import Input
# #
# # # Calculate class weights
# # import numpy as np
# # class_weights = {i: len(y)/(len(np.unique(y))*np.bincount(y)[i])
# #                  for i in np.unique(y)}
# #
# # # # Build model
# # # model = Sequential([
# # #     Dense(64, activation='relu', input_shape=(7,)),
# # #     Dropout(0.3),
# # #     Dense(32, activation='relu'),
# # #     Dense(len(le.classes_), activation='softmax')
# # # ])
# #
# # model = Sequential([
# #     Input(shape=(7,)),  # Input layer explicitly defined
# #     Dense(64, activation='relu'),
# #     Dropout(0.3),
# #     Dense(32, activation='relu'),
# #     Dense(len(le.classes_), activation='softmax')
# # ])
# #
# # model.compile(optimizer='adam',
# #               loss='sparse_categorical_crossentropy',
# #               metrics=['accuracy'])
# #
# # # Train with early stopping
# # early_stop = EarlyStopping(monitor='val_loss', patience=10)
# # history = model.fit(
# #     X_train_scaled, y_train,
# #     epochs=200,
# #     validation_split=0.2,
# #     class_weight=class_weights,
# #     callbacks=[early_stop],
# #     verbose=0
# # )
# #
# # # Evaluate
# # dl_loss, dl_acc = model.evaluate(X_test_scaled, y_test)
# # print(f"\nNeural Network Accuracy: {dl_acc:.2f}")
# #
# # # ====================
# # # Visualization
# # # ====================
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import confusion_matrix
# # import seaborn as sns
# #
# # # Feature importance
# # importances = rf_model.feature_importances_
# # features = X.columns
# # plt.figure(figsize=(10,6))
# # sns.barplot(x=importances, y=features)
# # plt.title('Feature Importance')
# # plt.show()
# #
# # # Confusion matrix
# # cm = confusion_matrix(y_test, y_pred)
# # plt.figure(figsize=(12,8))
# # sns.heatmap(cm, annot=True, fmt='d',
# #             xticklabels=le.classes_,
# #             yticklabels=le.classes_)
# # plt.title('Confusion Matrix')
# # plt.xlabel('Predicted')
# # plt.ylabel('Actual')
# # plt.show()
# #
# # # ====================
# # # Recommendation System
# # # ====================
# # def crop_recommender(N, P, K, temp, humidity, ph, rainfall):
# #     """
# #     Returns crop probabilities in descending order
# #     """
# #     inputs = scaler.transform([[N, P, K, temp, humidity, ph, rainfall]])
# #
# #     # Get predictions from both models
# #     rf_proba = rf_model.predict_proba(inputs)[0]
# #     dl_proba = model.predict(inputs, verbose=0)[0]
# #
# #     # Average probabilities
# #     avg_proba = (rf_proba + dl_proba) / 2
# #
# #     # Create sorted dictionary
# #     crops = le.inverse_transform(np.arange(len(avg_proba)))
# #     return {crop: f"{prob:.1%}" for crop, prob in
# #             sorted(zip(crops, avg_proba),
# #             key=lambda x: x[1], reverse=True)}
# #
# # # Example usage
# # print(crop_recommender(90, 42, 43, 21, 82, 6.5, 203))
# #
# # # TRY 2
# # # ====================
# # # Data Preparation
# # # ====================
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # import tensorflow as tf
# # # from tensorflow.keras.models import Sequential
# # # from tensorflow.keras.layers import Dense, Dropout
# # # from tensorflow.keras.callbacks import EarlyStopping
# # #
# # # # Load dataset from Excel file
# # # data = pd.read_excel('C:/Users/likul/Desktop/Project/AgriGuide/sampledata.xlsx')
# # #
# # # # Encode labels
# # # le = LabelEncoder()
# # # data['label'] = le.fit_transform(data['label'])
# # #
# # # # Split features and target
# # # X = data.drop('label', axis=1)
# # # y = data['label']
# # #
# # # # Split dataset (70-30 split)
# # # X_train, X_test, y_train, y_test = train_test_split(
# # #     X, y, test_size=0.3, random_state=42
# # # )
# # #
# # # # Standardize features
# # # scaler = StandardScaler()
# # # X_train_scaled = scaler.fit_transform(X_train)
# # # X_test_scaled = scaler.transform(X_test)
# # #
# # # # ====================
# # # # Machine Learning Model
# # # # ====================
# # # # Initialize and train Random Forest
# # # rf_model = RandomForestClassifier(
# # #     n_estimators=200,
# # #     max_depth=5,
# # #     class_weight='balanced',
# # #     random_state=42
# # # )
# # # rf_model.fit(X_train_scaled, y_train)
# # #
# # # # Evaluate
# # # y_pred = rf_model.predict(X_test_scaled)
# # # print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# # # print(classification_report(y_test, y_pred, target_names=le.classes_))
# # #
# # # # ====================
# # # # Deep Learning Model
# # # # ====================
# # # # Calculate class weights
# # # class_weights = {i: len(y) / (len(np.unique(y)) * np.bincount(y)[i])
# # #                  for i in np.unique(y)}
# # #
# # # # Build model
# # # model = Sequential([
# # #     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
# # #     Dropout(0.3),
# # #     Dense(32, activation='relu'),
# # #     Dense(len(le.classes_), activation='softmax')
# # # ])
# # #
# # # model.compile(optimizer='adam',
# # #               loss='sparse_categorical_crossentropy',
# # #               metrics=['accuracy'])
# # #
# # # # Train with early stopping
# # # early_stop = EarlyStopping(monitor='val_loss', patience=10)
# # # history = model.fit(
# # #     X_train_scaled, y_train,
# # #     epochs=200,
# # #     validation_split=0.2,
# # #     class_weight=class_weights,
# # #     callbacks=[early_stop],
# # #     verbose=0
# # # )
# # #
# # # # Evaluate
# # # dl_loss, dl_acc = model.evaluate(X_test_scaled, y_test)
# # # print(f"\nNeural Network Accuracy: {dl_acc:.2f}")
# # #
# # # # ====================
# # # # Visualization
# # # # ====================
# # # # Feature importance
# # # plt.figure(figsize=(10, 6))
# # # sns.barplot(x=rf_model.feature_importances_, y=X.columns)
# # # plt.title('Feature Importance')
# # # plt.show()
# # #
# # # # Confusion matrix
# # # plt.figure(figsize=(12, 8))
# # # sns.heatmap(confusion_matrix(y_test, y_pred),
# # #             annot=True, fmt='d',
# # #             xticklabels=le.classes_,
# # #             yticklabels=le.classes_)
# # # plt.title('Confusion Matrix')
# # # plt.xlabel('Predicted')
# # # plt.ylabel('Actual')
# # # plt.show()
# # #
# # #
# # # # ====================
# # # # Recommendation System
# # # # ====================
# # # def crop_recommender(N, P, K, temp, humidity, ph, rainfall):
# # #     """
# # #     Returns crop probabilities in descending order
# # #     """
# # #     inputs = scaler.transform([[N, P, K, temp, humidity, ph, rainfall]])
# # #
# # #     # Get predictions from both models
# # #     rf_proba = rf_model.predict_proba(inputs)[0]
# # #     dl_proba = model.predict(inputs, verbose=0)[0]
# # #
# # #     # Average probabilities
# # #     avg_proba = (rf_proba + dl_proba) / 2
# # #
# # #     # Create sorted dictionary
# # #     crops = le.inverse_transform(np.arange(len(avg_proba)))
# # #     return {crop: f"{prob:.1%}" for crop, prob in
# # #             sorted(zip(crops, avg_proba),
# # #                    key=lambda x: x[1], reverse=True)}
# # #
# # #
# # # # Example usage
# # # print("\nCrop Recommendation Example:")
# # # print(crop_recommender(90, 42, 43, 21, 82, 6.5, 203))