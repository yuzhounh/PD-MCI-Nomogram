import pandas as pd

# Read the CSV file.
df = pd.read_csv("PPMI_6_train.csv")

# Count missing values for each feature.
missing_count = df.isnull().sum()

# Calculate the missing percentage for each feature.
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Create a summary DataFrame.
missing_df = pd.DataFrame(
    {"Missing count": missing_count, "Missing percentage (%)": missing_percentage}
)

# Sort by missing percentage in descending order.
missing_df = missing_df.sort_values(by="Missing percentage (%)", ascending=False)

# Display the results.
print("=" * 60)
print("Feature missingness statistics (descending by missing percentage)")
print("=" * 60)
print(f"Total number of observations: {len(df)}")
print("-" * 60)
print(missing_df)
print("-" * 60)

# Display only features with missing values.
missing_features = missing_df[missing_df["Missing count"] > 0]
if len(missing_features) > 0:
    print(f"\nNumber of features with missing values: {len(missing_features)}")
else:
    print("\nNo features have missing values")
