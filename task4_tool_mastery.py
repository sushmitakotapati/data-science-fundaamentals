
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load a sample dataset
iris = load_iris(as_frame=True)
df = iris.frame

print("First five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# 2. Check and clean data
print("\nMissing values:")
print(df.isnull().sum())

df = df.drop_duplicates()

# 3. Use NumPy for numerical analysis
values = df["sepal length (cm)"].to_numpy()
print("\nMean sepal length:", np.mean(values))
print("Median sepal length:", np.median(values))

# 4. Visualize the data
plt.figure(figsize=(7, 5))
sns.histplot(df["sepal length (cm)"], kde=True)
plt.title("Distribution of Sepal Length")
plt.tight_layout()
plt.savefig("task4_histogram.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="target"
)
plt.title("Sepal Length vs Petal Length")
plt.tight_layout()
plt.savefig("task4_scatterplot.png")
plt.close()

# 5. Train a basic machine learning model
X = df[iris.feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nModel accuracy:", accuracy_score(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

print("\nTask 4 completed successfully.")
