import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Check for missing values
print("Original dataset shape: ", df.shape)
print("Number of missing or null values: ", df.isnull().sum())
df = df.dropna(how="any")
print("Dataset shape after removing missing values: ", df.shape)

# Dataset statistics
print("Dataset statistics: ")
print(df.describe())

# Preview dataset
print("Preview dataset: ")
print(df.head())

# Feature matrix and target vector
feature_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
target_name = "target"
X = df.loc[:, feature_names].values
y = df.loc[:, target_name].values

print("Feature matrix shape: ", X.shape)
print("Target vector shape: ", y.shape)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rf_classifier = RandomForestClassifier(
    n_estimators=100, max_depth=5, min_samples_split=2, random_state=0
)
rf_classifier.fit(X_train, y_train)

# Prediction
y_hat = rf_classifier.predict(X_test)

# Score
print("Score: ", rf_classifier.score(X_test, y_test))

# Evaluation
print("\nModel evaluation: ")
print("Accuracy: ", accuracy_score(y_test, y_hat))
print("\nClassification report: ")
print(classification_report(y_test, y_hat))
print("\nConfusion matrix: ")
print(confusion_matrix(y_test, y_hat))

# Feature importance
feature_importance = pd.DataFrame(
    {"feature": feature_names, "importance": rf_classifier.feature_importances_}
)
print("\nFeature importance: ")
print(feature_importance.sort_values(by="importance", ascending=False))

# Visualize feature importance
plt.figure(figsize=(12, 8))
plt.bar(feature_importance["feature"], feature_importance["importance"])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()