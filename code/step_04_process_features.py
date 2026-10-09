import pandas as pd
import numpy as np

print("\n=== Step 04: Process Features & Cohort Filtering (Stage 1) ===")

# Read the table.
df = pd.read_csv("PPMI_3_source.csv", low_memory=False)
initial_visits = len(df)
initial_subjects = df["PATNO"].nunique()
print(
    f"Initial records from PPMI_3_source: {initial_visits} visits, {initial_subjects} subjects"
)
if (initial_visits, initial_subjects) != (8546, 1684):
    raise RuntimeError(
        "Locked cohort entry changed: "
        f"expected 8546 visits/1684 subjects, got {initial_visits}/{initial_subjects}"
    )

# Protocol rule 1: Exclude visits with missing MCI_testscores (no backfilling or imputation).
if "MCI_testscores" in df.columns:
    missing_mci_mask = df["MCI_testscores"].isna()
    n_missing_mci = missing_mci_mask.sum()
    print(f"Excluded visits with missing MCI_testscores: {n_missing_mci}")
    df = df[~missing_mci_mask].copy()
print(
    f"After removing missing outcome: {len(df)} visits, {df['PATNO'].nunique()} subjects"
)
if (len(df), df["PATNO"].nunique()) != (7959, 1674):
    raise RuntimeError(
        "Locked missing-outcome exclusion changed: "
        f"expected 7959 visits/1674 subjects, got {len(df)}/{df['PATNO'].nunique()}"
    )

# Protocol rule 2: Exclude visits with a documented PDD diagnosis (cogstate == 3.0).
if "cogstate" in df.columns:
    pdd_mask = df["cogstate"] == 3.0
    n_pdd = pdd_mask.sum()
    pdd_subjects = df.loc[pdd_mask, "PATNO"].nunique()
    print(
        f"Excluded visits with PDD (cogstate == 3.0): {n_pdd} visits across {pdd_subjects} subjects"
    )
    df = df[~pdd_mask].copy()
print(f"After excluding PDD visits: {len(df)} visits, {df['PATNO'].nunique()} subjects")
if (len(df), df["PATNO"].nunique()) != (7817, 1673):
    raise RuntimeError(
        "Locked PDD visit exclusion changed: "
        f"expected 7817 visits/1673 subjects, got {len(df)}/{df['PATNO'].nunique()}"
    )

# Select features with flag equal to 1, preserving their original column order.
feature_sets = pd.read_csv("PPMI_feature_sets.csv")
selected_features = feature_sets[feature_sets["flag"] == 1]["feature"].tolist()
existing_selected_features = [col for col in df.columns if col in selected_features]
df = df[existing_selected_features].copy()

# Set BMI values above 45 to missing.
df.loc[df["BMI"] > 45, "BMI"] = pd.NA

# Set the age at onset for subject 41172 to missing.
# Clinical QC: Subject 41172 has age at diagnosis (agediag) 73.83 years and recorded age at onset (ageonset) 19.62 years.
# The difference exceeds 54 years, an implausible interval for idiopathic Parkinson disease (suspected PPMI entry error).
# Set it to NA for correction in step_06 using the cohort median difference between diagnosis and onset ages.
df.loc[df["PATNO"] == 41172, "ageonset"] = pd.NA

# Recode fampd_bin and td_pigd by mapping {1, 2} to {1, 0}.
df["fampd_bin"] = df["fampd_bin"].map({1: 1, 2: 0})
df["td_pigd"] = df["td_pigd"].map({1: 1, 2: 0})

# Create binary has_GBA and has_LRRK2 features.
df["has_GBA"] = df["subgroup"].str.contains("GBA", case=False, na=False).astype(int)
df["has_LRRK2"] = df["subgroup"].str.contains("LRRK2", case=False, na=False).astype(int)

# Move MCI_testscores to the last column and rename it MCI.
if "MCI_testscores" in df.columns:
    mci_col = df.pop("MCI_testscores")
    df["MCI"] = mci_col

# Ensure that MCI has no missing values.
df = df.dropna(subset=["MCI"]).copy()

# Report sample statistics.
print("\n=== Final Cohort Statistics (Locked) ===")
print(f"Final valid visits: {len(df)}")
print(f"Final unique subjects: {df['PATNO'].nunique()}")
print(f"Feature count: {len(df.columns) - 6}")
print("Label distribution (MCI):")
val_counts = df["MCI"].value_counts().to_dict()
val_props = df["MCI"].value_counts(normalize=True).to_dict()
for k in sorted(val_counts.keys()):
    print(f"  Class {int(k)}: {val_counts[k]} ({val_props[k]*100:.2f}%)")

# Save the results.
df.to_csv("PPMI_4_features.csv", index=False)
print(f"\nSaved PPMI_4_features.csv successfully with shape {df.shape}")
