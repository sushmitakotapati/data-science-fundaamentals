
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create an illustrative retail sales dataset
data = {
    "Month_Number": list(range(1, 13)),
    "Electronics": [
        12000, 15000, 18000, 16000, 22000, 25000,
        24000, 26000, 28000, 30000, 34000, 38000
    ],
    "Clothing": [
        8000, 9000, 11000, 13000, 14000, 16000,
        15000, 17000, 18000, 20000, 22000, 25000
    ]
}

df = pd.DataFrame(data)
df["Total_Sales"] = df["Electronics"] + df["Clothing"]
df.to_csv("capstone_sales_data.csv", index=False)

# 2. Inspect and prepare the data
print("DATA PREVIEW")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

# 3. Monthly sales visualization
plt.figure(figsize=(9, 5))
plt.plot(df["Month_Number"], df["Total_Sales"], marker="o")
plt.title("Monthly Retail Sales")
plt.xlabel("Month Number")
plt.ylabel("Total Sales")
plt.grid(True)
plt.tight_layout()
plt.savefig("capstone_monthly_sales.png")
plt.close()

# 4. Category comparison
category_totals = {
    "Electronics": df["Electronics"].sum(),
    "Clothing": df["Clothing"].sum()
}

plt.figure(figsize=(7, 5))
plt.bar(category_totals.keys(), category_totals.values())
plt.title("Sales by Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("capstone_category_sales.png")
plt.close()

# 5. Simple sales prediction
# This is an illustrative time-trend model, not a validated forecast.
X = df[["Month_Number"]]
y = df["Total_Sales"]

model = LinearRegression()
model.fit(X.iloc[:10], y.iloc[:10])

test_predictions = model.predict(X.iloc[10:])
actual_values = y.iloc[10:]

print("\nActual sales for months 11-12:")
print(actual_values.to_list())

print("\nPredicted sales for months 11-12:")
print(np.round(test_predictions, 2).tolist())

print("\nMean Absolute Error:",
      round(mean_absolute_error(actual_values, test_predictions), 2))

print("R-squared on two test months:",
      round(r2_score(actual_values, test_predictions), 3))

# 6. Actual versus predicted graph
plt.figure(figsize=(7, 5))
plt.plot([11, 12], actual_values, marker="o", label="Actual")
plt.plot([11, 12], test_predictions, marker="x", label="Predicted")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Month Number")
plt.ylabel("Total Sales")
plt.legend()
plt.tight_layout()
plt.savefig("capstone_actual_vs_predicted.png")
plt.close()

print("\nCapstone analysis completed successfully.")
print("Dataset and three graphs have been saved.")
