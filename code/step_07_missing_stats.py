import pandas as pd

# Read the CSV file.
df = pd.read_csv("PPMI_6_train.csv")

# Count missing values for each feature.
missing_count = df.isnull().sum()

# Calculate the missing percentage for each feature.
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Create a summary DataFrame.
missing_df = pd.DataFrame(
    {"缺失数量": missing_count, "缺失比例(%)": missing_percentage}
)

# Sort by missing percentage in descending order.
missing_df = missing_df.sort_values(by="缺失比例(%)", ascending=False)

# Display the results.
print("=" * 60)
print("特征缺失统计（按缺失比例从高到低排列）")
print("=" * 60)
print(f"总样本数: {len(df)}")
print("-" * 60)
print(missing_df)
print("-" * 60)

# Display only features with missing values.
missing_features = missing_df[missing_df["缺失数量"] > 0]
if len(missing_features) > 0:
    print(f"\n有缺失值的特征数量: {len(missing_features)}")
else:
    print("\n所有特征均无缺失值")
