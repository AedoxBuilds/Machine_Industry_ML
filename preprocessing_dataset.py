"""
Machine Failure Risk Prediction (Classification Version)
----------------------------------------------------------
Failure_Risk only ever takes 3 fixed values (e.g. 1.0, 0.0, -1.0),
which makes it a CATEGORICAL target, not a continuous number.
That means this is a classification problem, not a regression
problem -- so we use RandomForestClassifier instead of
LinearRegression.

This script:
1. Loads a CSV dataset
2. Cleans missing values
3. Encodes categorical columns (e.g. Machine_Type)
4. Scales numeric columns
5. Trains a classification model
6. Evaluates it with classification metrics (accuracy, confusion
   matrix, classification report) instead of regression metrics
   (MAE/MSE/R2), since those don't make sense for class labels
7. Predicts Failure_Risk for a brand new sample

The scaler and label encoders fitted during training are SAVED and
REUSED when predicting on new data, so the new sample is processed
in exactly the same way the training data was.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# ==========================================
# 1. Load the dataset
# ==========================================

file_name = "Machine_Industry.csv"
df = pd.read_csv(file_name)

print("Original Data:")
print(df)

print("\nColumns:")
print(df.columns)

# Check for missing values before cleaning
print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())


# ==========================================
# 2. Select the target column (what we want to predict)
# ==========================================

target_column = input("\nEnter target column name: ")

X = df.drop(target_column, axis=1)   # everything except the target = features
y = df[target_column]                # the column we're trying to predict


# ==========================================
# 3. Automatically detect numeric vs categorical FEATURE columns
# ==========================================
# Note: this is about the INPUT features (X), not the target (y).
# The target is handled separately in step 5b below.

numeric_columns = X.select_dtypes(include="number").columns
categorical_columns = X.select_dtypes(exclude="number").columns

print("\nNumeric Columns:", list(numeric_columns))
print("Categorical Columns:", list(categorical_columns))


# ==========================================
# 4. Handle missing values
# ==========================================

# Numeric columns -> fill missing values with the column's mean
for column in numeric_columns:
    X[column] = X[column].fillna(X[column].mean())

# Categorical columns -> fill missing values with the most common value (mode)
for column in categorical_columns:
    X[column] = X[column].fillna(X[column].mode()[0])


# ==========================================
# 5a. Encode categorical FEATURE columns into numbers
# ==========================================
# We store one LabelEncoder per categorical column in a dictionary,
# so the exact same encoding can be reused later on new data.

encoders = {}  # column_name -> fitted LabelEncoder

for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    encoders[column] = le   # save this encoder for later reuse


# ==========================================
# 5b. Prepare the TARGET column for classification
# ==========================================
# RandomForestClassifier works fine with numeric class labels like
# 1.0 / 0.0 / -1.0 directly, so no target encoding is strictly
# required. We just make sure it's treated as a category, not a
# float to be measured numerically.

print("\nUnique target classes found:", sorted(y.unique()))


# ==========================================
# 6. Scale numeric feature columns
# ==========================================
# Only the FEATURES (X) are scaled -- never the target (y), since
# scaling the target would distort the class labels themselves.
# We keep the fitted `scaler` object so the same scaling can be
# applied to new prediction data later.

scaler = MinMaxScaler(feature_range=(-1, 1))
X[X.columns] = scaler.fit_transform(X)


# ==========================================
# 7. Split into training and testing sets
# ==========================================
# stratify=y keeps the same proportion of each class in both the
# train and test sets -- important for classification, especially
# if one class (e.g. -1.0) is rarer than the others.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==========================================
# 8. Display results
# ==========================================

print("\n==============================")
print("Preprocessed Data")
print("==============================")
print(X)

print("\n==============================")
print("Training Data")
print("==============================")
print(X_train)

print("\n==============================")
print("Testing Data")
print("==============================")
print(X_test)


# ==========================================
# 9. Save the preprocessed dataset to CSV
# ==========================================

processed_df = X.copy()
processed_df[target_column] = y.values
processed_df.to_csv("preprocessed_data.csv", index=False)

print("\nPreprocessed data saved successfully!")


# ==========================================
# 10. Train a classification model
# ==========================================
# RandomForestClassifier handles mixed numeric/categorical-style
# data well, needs no scaling to work internally, and gives us
# feature importances for free.

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)


# ==========================================
# 11. Evaluate with CLASSIFICATION metrics
# ==========================================
# MAE / MSE / R2 don't make sense for class labels -- they measure
# distance between numbers, but "how far" 1.0 is from 0.0 has no
# real meaning here. Instead we use:
#   - Accuracy: % of predictions that were exactly correct
#   - Confusion matrix: breakdown of correct vs incorrect predictions
#     per class
#   - Classification report: precision/recall/F1 per class

print("\n==============================")
print("Model Performance on Test Data")
print("==============================")
print("Accuracy:", accuracy_score(y_test, y_pred_test))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Bonus: which features mattered most to the model
print("\nFeature Importances:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"  {feature}: {importance:.4f}")


# ==========================================
# 11b. Visualize model performance
# ==========================================
# Two plots side by side:
#   Left  -> Confusion Matrix: how often predictions matched the
#            actual class (diagonal = correct, off-diagonal = errors)
#   Right -> Feature Importance: which input columns the model
#            relied on most to make its predictions

from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(12, 5))

# Left plot: Confusion Matrix
plt.subplot(1, 2, 1)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=plt.gca())
plt.title("Confusion Matrix — Actual vs Predicted")

# Right plot: Feature Importance
plt.subplot(1, 2, 2)
plt.bar(X.columns, model.feature_importances_, color="steelblue")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# ==========================================
# 12. Predict on a brand new sample
# ==========================================
# Enter RAW values here, exactly as they'd appear in the original
# CSV (e.g. the real category name for Machine_Type, like "Mill",
# NOT a guessed number). The code below encodes and scales it
# automatically using the SAME encoder/scaler fitted on training data.

new_sample_raw = {
    "Temperature": 19,
    "Vibration": 80,
    "Power_Usage": 19,
    "Humidity": 11,
    "Machine_Type": "Mill",  # must match one of encoders["Machine_Type"].classes_
}

new_sample = pd.DataFrame([new_sample_raw])

# Make sure the new sample's columns are in the SAME order as X
new_sample = new_sample[X.columns]

# Apply the same label encoding used during training
for column in categorical_columns:
    new_sample[column] = encoders[column].transform(
        new_sample[column].astype(str)
    )

# Apply the same scaling used during training
new_sample[new_sample.columns] = scaler.transform(new_sample)

# Predict the class label
predicted_class = model.predict(new_sample)[0]

# Predict the model's confidence (probability) for each class
predicted_probabilities = model.predict_proba(new_sample)[0]

print("\n==============================")
print("Prediction for New Sample")
print("==============================")
print("Predicted Failure Risk class:", predicted_class)
print("Class probabilities:")
for class_label, probability in zip(model.classes_, predicted_probabilities):
    print(f"  {class_label}: {probability:.2%}")