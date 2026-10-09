import pandas as pd


RAW_DATA_FILE = "PPMI_Curated_Data_Cut_Public_20260810.xlsx"
EXPECTED_PD_COHORT = (8546, 1684)

# The 2026-08-10 curated cut renamed four DATSCAN columns. Keep the established
# downstream feature names so the remaining pipeline and manuscript tables use a
# stable interface.
SOURCE_COLUMN_ALIASES = {
    "MIA_LOWPUT_EXPECTED": "mia_lowput_expected",
    "MIA_CAUDATE_BILAT": "MIA_CAUDATE_mean",
    "MIA_PUTAMEN_BILAT": "MIA_PUTAMEN_mean",
    "MIA_STRIATUM_BILAT": "MIA_STRIATUM_mean",
}


def normalize_source_columns(df):
    rename_map = {}
    for source_name, canonical_name in SOURCE_COLUMN_ALIASES.items():
        if source_name in df.columns and canonical_name in df.columns:
            raise RuntimeError(
                f"Raw data contains both {source_name} and {canonical_name}; "
                "the DATSCAN alias is ambiguous"
            )
        if source_name in df.columns:
            rename_map[source_name] = canonical_name
    return df.rename(columns=rename_map)


print("\n开始处理PPMI数据...")

# Step 1: Read the Excel file and save it as CSV.
print("\n步骤1: 读取Excel文件...")
# Read the Excel file.
df_original = pd.read_excel(RAW_DATA_FILE)
print(f"原始数据形状: {df_original.shape}")

# Save as CSV.
df_original.to_csv("PPMI_0_data.csv", index=False)
print(f"已保存 PPMI_0_data.csv, 形状: {df_original.shape}")

# Normalize only confirmed upstream field renames; PPMI_0_data.csv retains the original column names.
df_source = normalize_source_columns(df_original)

# Step 2: Select rows with COHORT equal to 1.
print("\n步骤2: 筛选COHORT列为1的行...")
# Select rows with COHORT equal to 1.
df_pd = df_source[df_source["COHORT"] == 1].copy()
print(f"筛选后数据形状: {df_pd.shape}")

# Report the number of subjects (unique PATNO values).
num_subjects = df_pd["PATNO"].nunique()
print(f"被试（PATNO唯一数）数量: {num_subjects}")
if (len(df_pd), num_subjects) != EXPECTED_PD_COHORT:
    raise RuntimeError(
        "Locked initial PD cohort changed: "
        f"expected {EXPECTED_PD_COHORT[0]} visits/{EXPECTED_PD_COHORT[1]} subjects, "
        f"got {len(df_pd)} visits/{num_subjects} subjects"
    )

# Save the filtered data.
df_pd.to_csv("PPMI_1_PD.csv", index=False)
print(f"已保存 PPMI_1_PD.csv, 形状: {df_pd.shape}")

# Step 3: Read the risk-factor configuration.
print("\n步骤3: 处理风险因子特征...")
# Read the risk-factor configuration.
df_risk_config = pd.read_csv("PPMI_initial_feature_sets.csv")
print(f"风险因子配置文件形状: {df_risk_config.shape}")
print("配置文件前几行:")
print(df_risk_config.head())

# List features with flag equal to 1.
selected_features = df_risk_config[df_risk_config["flag"] == 1]["feature"].tolist()
print(f"选中的特征数量: {len(selected_features)}")
print(f"选中的特征: {selected_features[:10]}...")  # Show the first 10 features.

# Check whether these features exist in the data.
available_features = [feat for feat in selected_features if feat in df_pd.columns]
missing_features = [feat for feat in selected_features if feat not in df_pd.columns]

print(f"数据中可用的特征数量: {len(available_features)}")
if missing_features:
    raise RuntimeError(
        "Required features are missing after source-column normalization: "
        f"{missing_features}"
    )

# Extract the selected features.
df_risk_selected = df_pd[available_features].copy()
print(f"提取风险因子后数据形状: {df_risk_selected.shape}")

# Save the results.
df_risk_selected.to_csv("PPMI_2_features.csv", index=False)
print(f"已保存 PPMI_2_features.csv, 形状: {df_risk_selected.shape}")

# Save an identical copy under the existing downstream input filename, without cross-table backfilling.
df_risk_selected.to_csv("PPMI_3_source.csv", index=False)
print(f"已保存 PPMI_3_source.csv（curated 数据副本）, 形状: {df_risk_selected.shape}")

print("\n数据处理完成!")
print("=" * 50)
print("总结:")
print(f"1. PPMI_0_data.csv: {df_original.shape}")
print(f"2. PPMI_1_PD.csv: {df_pd.shape}")
print(f"3. PPMI_2_features.csv: {df_risk_selected.shape}")
print(f"4. PPMI_3_source.csv: {df_risk_selected.shape}")
