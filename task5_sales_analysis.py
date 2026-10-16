
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample retail sales dataset
data = {
    "Month": [
        "January", "January", "February", "February",
        "March", "March", "April", "April",
        "May", "May", "June", "June"
    ],
    "Category": [
        "Electronics", "Clothing", "Electronics", "Clothing",
        "Electronics", "Clothing", "Electronics", "Clothing",
        "Electronics", "Clothing", "Electronics", "Clothing"
    ],
    "Revenue": [
        12000, 8000, 15000, 9000,
        18000, 11000, 16000, 13000,
        22000, 14000, 25000, 16000
    ]
}

df = pd.DataFrame(data)
df.to_csv("sales_data.csv", index=False)

print("SALES DATA")
print(df)

monthly_sales = df.groupby("Month")["Revenue"].sum()
category_sales = df.groupby("Category")["Revenue"].sum()

print("\nMONTHLY SALES")
print(monthly_sales)

print("\nCATEGORY SALES")
print(category_sales)

print("\nTOTAL REVENUE:", df["Revenue"].sum())
print("AVERAGE REVENUE PER RECORD:", df["Revenue"].mean())

# Monthly sales graph
monthly_sales.plot(kind="bar", title="Monthly Retail Sales")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("monthly_sales.png")
plt.close()

# Category sales graph
category_sales.plot(kind="bar", title="Sales by Product Category")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("category_sales.png")
plt.close()

print("\nAnalysis completed. Graphs and CSV saved.")
