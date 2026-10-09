import pandas as pd

# Read training and test set data
train_df = pd.read_csv("PPMI_7_train.csv")
test_df = pd.read_csv("PPMI_7_test.csv")

# Remove columns
columns_to_remove = [
    "agediag",
    "ageonset",
    "updrs2_score",
    "stai_trait",
    "stai_state",
    "LEDD",
]

train_df_new = train_df.drop(columns=columns_to_remove)
test_df_new = test_df.drop(columns=columns_to_remove)

# Save to new files
train_df_new.to_csv("PPMI_8_train.csv", index=False)
test_df_new.to_csv("PPMI_8_test.csv", index=False)

# Display operation results
print(f"Training set original column count: {len(train_df.columns)}")
print(f"Training set new column count: {len(train_df_new.columns)}")
print(f"Training set row count: {len(train_df_new)}")

print(f"\nTest set original column count: {len(test_df.columns)}")
print(f"Test set new column count: {len(test_df_new.columns)}")
print(f"Test set row count: {len(test_df_new)}")

print("\nRemoved columns:", columns_to_remove)
