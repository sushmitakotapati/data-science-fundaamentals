
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create a folder for results
output_dir = Path("task3_outputs")
output_dir.mkdir(exist_ok=True)

# 1. Load the dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
)

print("PALMER PENGUINS DATA ANALYSIS")
print("=" * 40)

# 2. Inspect the dataset
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# 3. Check data quality
print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Save a copy after removing duplicate rows
df = df.drop_duplicates().copy()

# 4. Summary statistics
print("\nNumerical summary:")
print(df.describe())

print("\nNumber of records by species:")
print(df["species"].value_counts())

print("\nAverage measurements by species:")
species_summary = df.groupby("species")[
    ["bill_length_mm", "bill_depth_mm",
     "flipper_length_mm", "body_mass_g"]
].mean().round(2)

print(species_summary)
species_summary.to_csv(output_dir / "species_summary.csv")

# 5. Missing-value handling for analysis
# Fill missing numerical values with each column's median.
analysis_df = df.copy()

numeric_cols = analysis_df.select_dtypes(
    include="number"
).columns

for col in numeric_cols:
    analysis_df[col] = analysis_df[col].fillna(
        analysis_df[col].median()
    )

# Fill missing categories with their most frequent value.
for col in analysis_df.select_dtypes(
    include=["object", "category"]
).columns:
    if analysis_df[col].isnull().any():
        mode_values = analysis_df[col].mode()
        if not mode_values.empty:
            analysis_df[col] = analysis_df[col].fillna(
                mode_values.iloc[0]
            )

print("\nMissing values after cleaning:")
print(analysis_df.isnull().sum())

# 6. Graph 1: Count of penguins by species
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="species")
plt.title("Number of Penguins by Species")
plt.xlabel("Species")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(output_dir / "species_count.png", dpi=150)
plt.close()

# 7. Graph 2: Flipper length by species
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="species", y="flipper_length_mm")
plt.title("Flipper Length by Penguin Species")
plt.xlabel("Species")
plt.ylabel("Flipper Length (mm)")
plt.tight_layout()
plt.savefig(output_dir / "flipper_length.png", dpi=150)
plt.close()

# 8. Graph 3: Body mass vs flipper length
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="flipper_length_mm",
    y="body_mass_g",
    hue="species"
)
plt.title("Body Mass vs Flipper Length")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.tight_layout()
plt.savefig(output_dir / "body_mass_vs_flipper.png", dpi=150)
plt.close()

# 9. Graph 4: Correlation heatmap
plt.figure(figsize=(8, 5))
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
plt.close()

# 10. Save cleaned dataset
analysis_df.to_csv(
    output_dir / "penguins_cleaned.csv",
    index=False
)

# 11. Generate observations using actual dataset results
print("\nKEY OBSERVATIONS")
print("=" * 40)

print("Records after removing duplicates:", len(df))
print("Species included:", df["species"].nunique())
print("Islands included:", df["island"].nunique())

print("\nAverage body mass by species (grams):")
print(df.groupby("species")["body_mass_g"].mean().round(2))

print("\nAverage flipper length by species (mm):")
print(df.groupby("species")["flipper_length_mm"].mean().round(2))

print("\nFiles saved in:", output_dir.resolve())
print("Analysis completed successfully!")
