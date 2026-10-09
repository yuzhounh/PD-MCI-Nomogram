import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("PPMI_6_train.csv")
test_df = pd.read_csv("PPMI_6_test.csv")

print("=" * 80)
print("DATA LOADING COMPLETED")
print("=" * 80)
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print()

# Features not to impute
exclude_features = [
    "SITE",
    "PATNO",
    "subgroup",
    "EVENT_ID",
    "visit_date",
    "age_at_visit",
    "ageonset",
    "agediag",
    "duration",
    "MCI",  # Never include the outcome label in imputation, to prevent information leakage.
]

# ============================================================================
# TIME FEATURE ENGINEERING
# ============================================================================


def add_comprehensive_time_features(df, dataset_name=""):
    """
    Add multiple time features for accurate temporal modeling

    Creates:
    1. months_from_baseline: Standard months based on EVENT_ID
    2. actual_months_from_baseline: Actual months from visit_date
    """
    prefix = f"[{dataset_name}] " if dataset_name else ""
    print(f"{prefix}\nAdding comprehensive time features...")

    # 1. Standard months based on EVENT_ID (with non-uniform intervals)
    event_to_months = {
        "BL": 0,
        "V01": 3,
        "V02": 6,
        "V03": 9,
        "V04": 12,
        "V05": 18,
        "V06": 24,
        "V07": 30,
        "V08": 36,
        "V09": 42,
        "V10": 48,
        "V11": 54,
        "V12": 60,
        "V13": 72,
        "V14": 84,
        "V15": 96,
        "V16": 108,
        "V17": 120,
        "V18": 132,
        "V19": 144,
    }
    df["months_from_baseline"] = df["EVENT_ID"].map(event_to_months)

    print(
        f"  - Mapped EVENT_ID to standard months (0 to {df['months_from_baseline'].max()})"
    )

    # 2. Convert visit_date to datetime BEFORE sorting
    df["visit_date_dt"] = pd.to_datetime(
        df["visit_date"], format="%m/%Y", errors="coerce"
    )

    # 3. Sort by PATNO and actual date (not string)
    df = df.sort_values(["PATNO", "visit_date_dt"]).reset_index(drop=True)

    # 4. Get baseline date for each patient
    baseline_dates = df[df["EVENT_ID"] == "BL"].set_index("PATNO")["visit_date_dt"]
    df["baseline_date_dt"] = df["PATNO"].map(baseline_dates)

    # 5. Calculate actual months (more precise than EVENT_ID)
    valid_dates = df["visit_date_dt"].notna() & df["baseline_date_dt"].notna()
    df.loc[valid_dates, "actual_months_from_baseline"] = (
        df.loc[valid_dates, "visit_date_dt"] - df.loc[valid_dates, "baseline_date_dt"]
    ).dt.days / 30.44

    # Fill missing actual_months with standard months
    df["actual_months_from_baseline"] = df["actual_months_from_baseline"].fillna(
        df["months_from_baseline"]
    )

    print(f"  - Calculated actual months from visit dates")
    print(f"    (accounts for individual visit timing variations)")

    # Clean up temporary columns (keep visit_date as string)
    df = df.drop(columns=["visit_date_dt", "baseline_date_dt"], errors="ignore")

    print()

    return df


print("=" * 80)
print("TIME FEATURE ENGINEERING")
print("=" * 80)

train_df = add_comprehensive_time_features(train_df, "Training Set")
test_df = add_comprehensive_time_features(test_df, "Test set")

# ============================================================================
# AUTO FEATURE TYPE DETECTION
# ============================================================================


def detect_feature_type(series, unique_threshold=6):
    """Automatically detect if a feature is continuous or categorical."""
    valid_values = series.dropna()

    if len(valid_values) == 0:
        return "continuous"

    n_unique = valid_values.nunique()

    is_all_int = False
    if valid_values.dtype in ["int64", "int32", "float64", "float32"]:
        is_all_int = np.all(valid_values == valid_values.astype(int))

    if n_unique <= unique_threshold or (is_all_int and n_unique <= 6):
        return "categorical"
    else:
        return "continuous"


print("=" * 80)
print("AUTOMATIC FEATURE TYPE DETECTION")
print("=" * 80)

continuous_features = []
categorical_features = []

# Exclude new time features from auto-detection
time_derived_features = ["months_from_baseline", "actual_months_from_baseline"]

for col in train_df.columns:
    if col in exclude_features + time_derived_features:
        continue

    feature_type = detect_feature_type(train_df[col])

    if feature_type == "continuous":
        continuous_features.append(col)
    else:
        categorical_features.append(col)

print(f"Total continuous features: {len(continuous_features)}")
print(f"Total categorical features: {len(categorical_features)}")
print()

# ============================================================================
# FEATURE CLASSIFICATION BY TEMPORAL CHARACTERISTICS
# ============================================================================

print("=" * 80)
print("FEATURE CLASSIFICATION BY TEMPORAL CHARACTERISTICS")
print("=" * 80)

# Static features (historical facts, never change)
static_features = [
    "EDUCYRS",
    "SEX",
    "fampd_bin",
    "has_GBA",
    "has_LRRK2",
    "APOE_e4",
    "sym_tremor",
    "sym_rigid",
    "sym_brady",
    "sym_posins",
    "sym_other",
]

# Monotonic features (only increase with time)
monotonic_features = []

# Dynamic continuous features (can change in both directions)
dynamic_continuous = [
    f for f in continuous_features if f not in static_features + monotonic_features
]

# Dynamic categorical features (can change states)
dynamic_categorical = [f for f in categorical_features if f not in static_features]

print(f"Static features: {len(static_features)}")
print(f"  {static_features}")
print(f"Monotonic features: {len(monotonic_features)}")
print(f"  {monotonic_features}")
print(f"Dynamic continuous features: {len(dynamic_continuous)}")
print(f"  {dynamic_continuous}")
print(f"Dynamic categorical features: {len(dynamic_categorical)}")
print(f"  {dynamic_categorical}")
print()

# ============================================================================
# MISSING RATE ANALYSIS
# ============================================================================

missing_counts = train_df.isnull().sum()
missing_rates = (missing_counts / len(train_df) * 100).sort_values(ascending=False)

test_missing_counts = test_df.isnull().sum()
test_missing_rates = test_missing_counts / len(test_df) * 100

# Output header with reduced column width and spacing
print(
    f"{'Feature':<14} {'#Train Miss':>12} {'%Train Miss':>12} {'#Test Miss':>12} {'%Test Miss':>12}"
)
print(f"{'-'*14} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

for feature in train_df.columns:
    if feature in exclude_features + time_derived_features:
        continue
    tr_cnt = missing_counts.get(feature, 0)
    tr_rate = missing_rates.get(feature, 0)
    te_cnt = test_missing_counts.get(feature, 0)
    te_rate = test_missing_rates.get(feature, 0)
    if tr_cnt > 0 or te_cnt > 0:
        print(
            f"{feature:<14} {tr_cnt:>12} {tr_rate:>12.2f} {te_cnt:>12} {te_rate:>12.2f}"
        )
print()

# Separate features by missing rate threshold
# Include any features that have missing values in train or test set
has_missing = [
    f
    for f in continuous_features + categorical_features
    if (train_df[f].isnull().sum() > 0) or (test_df[f].isnull().sum() > 0)
]

low_missing_features = [f for f in has_missing if missing_rates.get(f, 0) < 10]
high_missing_features = [f for f in has_missing if missing_rates.get(f, 0) >= 10]

print("=" * 80)
print("FEATURE CATEGORIZATION BY MISSING RATE")
print("=" * 80)
print(f"Low missing rate features (<10%): {len(low_missing_features)}")
print(f"High missing rate features (>=10%): {len(high_missing_features)}")
print()

# ============================================================================
# HELPER FUNCTIONS FOR INTELLIGENT IMPUTATION
# ============================================================================


def history_only_within_subject_imputation(df, feature, feature_type="continuous"):
    """
    For both training and test sets, use only information available at or before the target visit.

    - static: Forward fill (ffill) only; never backfill (bfill) from future visits.
    - monotonic: Interpolate using current and earlier observations only, without extrapolation; ffill only.
    - dynamic_continuous: Interpolate using current and earlier observations only; ffill only.
    - dynamic_categorical: ffill only (no bfill).
    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values(["PATNO", "actual_months_from_baseline"])

    if feature_type == "static":
        # Static features: Respect temporal constraints with ffill only; never bfill from future visits.
        df_copy[feature] = df_copy.groupby("PATNO")[feature].ffill()

    elif feature_type == "monotonic":
        # Monotonic features: Use historical observations only for linear interpolation/extrapolation; ffill only.
        for patno in df_copy["PATNO"].unique():
            mask = df_copy["PATNO"] == patno
            subject_data = df_copy.loc[mask].copy()

            non_missing = subject_data[feature].notna()

            if non_missing.sum() >= 2:
                x = subject_data.loc[non_missing, "actual_months_from_baseline"].values
                y = subject_data.loc[non_missing, feature].values

                # For each missing time point, use only observations available at or before that time.
                missing_mask = subject_data[feature].isna()
                if missing_mask.sum() > 0:
                    for idx in subject_data[missing_mask].index:
                        t_current = subject_data.loc[idx, "actual_months_from_baseline"]
                        # Select observations with time <= the current time point.
                        past_mask = x <= t_current
                        if past_mask.sum() >= 2:
                            f = interp1d(
                                x[past_mask],
                                y[past_mask],
                                kind="linear",
                                fill_value="extrapolate",
                            )
                            df_copy.loc[idx, feature] = f(t_current)
                        elif past_mask.sum() == 1:
                            # With only one historical observation, use that value directly.
                            df_copy.loc[idx, feature] = y[past_mask][0]

            elif non_missing.sum() == 1:
                # With only one value, forward fill only.
                df_copy.loc[mask, feature] = df_copy.loc[mask, feature].ffill()

    elif feature_type == "dynamic_continuous":
        # Dynamic continuous features: Interpolate using historical observations only; ffill only.
        for patno in df_copy["PATNO"].unique():
            mask = df_copy["PATNO"] == patno
            subject_data = df_copy.loc[mask].copy()

            non_missing = subject_data[feature].notna()

            if non_missing.sum() >= 2:
                x = subject_data.loc[non_missing, "actual_months_from_baseline"].values
                y = subject_data.loc[non_missing, feature].values

                # For each missing time point, interpolate using only observations available at or before that time.
                missing_mask = subject_data[feature].isna()
                if missing_mask.sum() > 0:
                    for idx in subject_data[missing_mask].index:
                        t_current = subject_data.loc[idx, "actual_months_from_baseline"]
                        past_mask = x <= t_current
                        if past_mask.sum() >= 2:
                            f = interp1d(
                                x[past_mask],
                                y[past_mask],
                                kind="linear",
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                            interpolated = f(t_current)
                            if not np.isnan(interpolated):
                                df_copy.loc[idx, feature] = interpolated

                # For remaining missing values, forward fill only.
                df_copy.loc[mask, feature] = df_copy.loc[mask, feature].ffill()

            elif non_missing.sum() == 1:
                df_copy.loc[mask, feature] = df_copy.loc[mask, feature].ffill()

    elif feature_type == "dynamic_categorical":
        # Categorical features: Forward fill only (states tend to persist); never backfill from future information.
        df_copy[feature] = df_copy.groupby("PATNO")[feature].ffill()

    return df_copy[feature]


def assign_visit_stage(months):
    """
    Assign visit stage based on actual months from baseline.
    More refined stage classification reflecting PD disease progression.
    """
    if pd.isna(months):
        return "unknown"
    elif months < 12:  # <1 year: Honeymoon phase.
        return "early"
    elif months < 36:  # 1-3 years: Stable symptom phase.
        return "mid_early"
    elif months < 60:  # 3-5 years: Complications begin to emerge.
        return "mid_late"
    elif months < 84:  # 5-7 years: Progression phase.
        return "late"
    else:  # >7 years: Advanced phase.
        return "very_late"


# Define all visit stages for stratified iteration.
VISIT_STAGES = ["early", "mid_early", "mid_late", "late", "very_late"]


def stratified_temporal_imputation(train_df, test_df, feature, is_categorical=False):
    """
    Stratified imputation considering temporal, demographic, and clinical factors
    Uses ACTUAL_MONTHS_FROM_BASELINE for accurate temporal stratification

    Returns:
    - imputation_params: dictionary with stratified statistics
    """
    train_copy = train_df.copy()

    # Set minimum sample sizes according to feature type.
    # Categorical features: Stable mode estimation requires a minimum number of observations per class.
    # Continuous features: Stable median estimation requires more observations.
    if is_categorical:
        n_classes = train_copy[feature].nunique()
        min_samples = max(5, n_classes * 2)  # At least 2 observations per class.
    else:
        min_samples = 10

    # Create age groups
    train_copy["age_group"] = pd.cut(
        train_copy["age_at_visit"],
        bins=[0, 60, 70, 100],
        labels=["young", "middle", "old"],
    )

    # Create duration groups (reference to code 1)
    train_copy["duration_group"] = pd.cut(
        train_copy["duration"], bins=[-1, 3, 7, 100], labels=["early", "mid", "late"]
    )

    # Create visit stage groups based on ACTUAL MONTHS (more accurate)
    train_copy["visit_stage"] = train_copy["actual_months_from_baseline"].apply(
        assign_visit_stage
    )

    # Build hierarchical stratification (reference to code 1)
    imputation_params = {}

    # Level 1: Most granular (subgroup + visit_stage + age_group + duration_group)
    for subgroup in train_copy["subgroup"].unique():
        for visit_stage in VISIT_STAGES:
            for age_group in ["young", "middle", "old"]:
                for duration_group in ["early", "mid", "late"]:
                    mask = (
                        (train_copy["subgroup"] == subgroup)
                        & (train_copy["visit_stage"] == visit_stage)
                        & (train_copy["age_group"] == age_group)
                        & (train_copy["duration_group"] == duration_group)
                    )

                    stratum_data = train_copy.loc[mask, feature].dropna()

                    if len(stratum_data) >= min_samples:
                        if is_categorical:
                            value = (
                                stratum_data.mode()[0]
                                if len(stratum_data.mode()) > 0
                                else None
                            )
                        else:
                            value = stratum_data.median()

                        key = (subgroup, visit_stage, age_group, duration_group)
                        imputation_params[key] = value

    # Level 2: Relax duration (subgroup + visit_stage + age_group)
    for subgroup in train_copy["subgroup"].unique():
        for visit_stage in VISIT_STAGES:
            for age_group in ["young", "middle", "old"]:
                key_l2 = (subgroup, visit_stage, age_group, "all")
                if key_l2 not in imputation_params:
                    mask = (
                        (train_copy["subgroup"] == subgroup)
                        & (train_copy["visit_stage"] == visit_stage)
                        & (train_copy["age_group"] == age_group)
                    )

                    stratum_data = train_copy.loc[mask, feature].dropna()

                    if len(stratum_data) >= min_samples:
                        if is_categorical:
                            value = (
                                stratum_data.mode()[0]
                                if len(stratum_data.mode()) > 0
                                else None
                            )
                        else:
                            value = stratum_data.median()
                        imputation_params[key_l2] = value

    # Level 3: Relax age (subgroup + visit_stage)
    for subgroup in train_copy["subgroup"].unique():
        for visit_stage in VISIT_STAGES:
            key_l3 = (subgroup, visit_stage, "all", "all")
            if key_l3 not in imputation_params:
                mask = (train_copy["subgroup"] == subgroup) & (
                    train_copy["visit_stage"] == visit_stage
                )

                stratum_data = train_copy.loc[mask, feature].dropna()

                if len(stratum_data) >= min_samples:
                    if is_categorical:
                        value = (
                            stratum_data.mode()[0]
                            if len(stratum_data.mode()) > 0
                            else None
                        )
                    else:
                        value = stratum_data.median()
                    imputation_params[key_l3] = value

    # Level 4: Most relaxed (subgroup only)
    for subgroup in train_copy["subgroup"].unique():
        key_l4 = (subgroup, "all", "all", "all")
        if key_l4 not in imputation_params:
            mask = train_copy["subgroup"] == subgroup
            stratum_data = train_copy.loc[mask, feature].dropna()

            if len(stratum_data) > 0:
                if is_categorical:
                    value = (
                        stratum_data.mode()[0] if len(stratum_data.mode()) > 0 else None
                    )
                else:
                    value = stratum_data.median()
                imputation_params[key_l4] = value

    # Level 5: Global fallback
    global_data = train_copy[feature].dropna()
    if len(global_data) > 0:
        if is_categorical:
            global_value = global_data.mode()[0] if len(global_data.mode()) > 0 else 0
        else:
            global_value = global_data.median()
        imputation_params[("all", "all", "all", "all")] = global_value

    return imputation_params


def apply_stratified_imputation(
    df, feature, imputation_params, is_categorical=False, is_static=False
):
    """
    Apply stratified imputation parameters to fill missing values
    Uses ACTUAL_MONTHS_FROM_BASELINE for accurate temporal stratification

    For static features (is_static=True), all missing values for the same patient
    are filled with the same value to ensure within-subject consistency.
    """
    df_copy = df.copy()

    # Create grouping variables (reference to code 1)
    df_copy["age_group"] = pd.cut(
        df_copy["age_at_visit"],
        bins=[0, 60, 70, 100],
        labels=["young", "middle", "old"],
    )
    df_copy["duration_group"] = pd.cut(
        df_copy["duration"], bins=[-1, 3, 7, 100], labels=["early", "mid", "late"]
    )

    # Assign visit stage based on ACTUAL MONTHS
    df_copy["visit_stage"] = df_copy["actual_months_from_baseline"].apply(
        assign_visit_stage
    )

    if is_static:
        # For static features, process by patient to ensure same value for all visits
        for patno in df_copy["PATNO"].unique():
            patno_mask = df_copy["PATNO"] == patno
            patno_missing = df_copy.loc[patno_mask, feature].isna()

            # If this patient has missing values to fill
            if patno_missing.any():
                # Use stratification info from patient's first row to determine fill value
                first_idx = df_copy.loc[patno_mask].index[0]
                row = df_copy.loc[first_idx]
                subgroup = row["subgroup"]
                visit_stage = row["visit_stage"]
                age_group = row["age_group"]
                duration_group = row["duration_group"]

                # Try hierarchical lookup (reference to code 1)
                keys_to_try = [
                    (subgroup, visit_stage, age_group, duration_group),
                    (subgroup, visit_stage, age_group, "all"),
                    (subgroup, visit_stage, "all", "all"),
                    (subgroup, "all", "all", "all"),
                    ("all", "all", "all", "all"),
                ]

                fill_value = None
                for key in keys_to_try:
                    if key in imputation_params:
                        fill_value = imputation_params[key]
                        break

                # Fill all missing rows for this patient with the same value
                if fill_value is not None:
                    missing_indices = df_copy.loc[
                        patno_mask & df_copy[feature].isna()
                    ].index
                    df_copy.loc[missing_indices, feature] = fill_value
    else:
        # For non-static features, fill row by row (reference to code 1)
        missing_mask = df_copy[feature].isna()

        for idx in df_copy[missing_mask].index:
            row = df_copy.loc[idx]
            subgroup = row["subgroup"]
            visit_stage = row["visit_stage"]
            age_group = row["age_group"]
            duration_group = row["duration_group"]

            # Try hierarchical lookup (reference to code 1)
            keys_to_try = [
                (subgroup, visit_stage, age_group, duration_group),
                (subgroup, visit_stage, age_group, "all"),
                (subgroup, visit_stage, "all", "all"),
                (subgroup, "all", "all", "all"),
                ("all", "all", "all", "all"),
            ]

            for key in keys_to_try:
                if key in imputation_params:
                    df_copy.loc[idx, feature] = imputation_params[key]
                    break

    return df_copy[feature]


# ============================================================================
# STEP 1: INTELLIGENT IMPUTATION FOR LOW MISSING RATE FEATURES (<10%)
# ============================================================================

print("=" * 80)
print("STEP 1: INTELLIGENT IMPUTATION (Missing Rate < 10%)")
print("=" * 80)

train_filled = train_df.copy()
test_filled = test_df.copy()

# Store stratified imputation parameters.
stratified_params = {}

print("\nPhase 1: Within-subject intelligent imputation...")
print("(Using actual_months_from_baseline for accurate temporal interpolation)")
print()
print(
    f"  {'Feature':<20} {'Train imputed':<15} {'Test imputed':<15} {'Train remaining':<17} {'Test remaining':<15}"
)
print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*17} {'-'*15}")

for feature in low_missing_features:
    # Count missing values before imputation.
    train_missing_before = train_filled[feature].isnull().sum()
    test_missing_before = test_filled[feature].isnull().sum()

    # Determine the feature type.
    if feature in static_features:
        ftype = "static"
    elif feature in monotonic_features:
        ftype = "monotonic"
    elif feature in dynamic_continuous:
        ftype = "dynamic_continuous"
    elif feature in dynamic_categorical:
        ftype = "dynamic_categorical"
    else:
        ftype = (
            "dynamic_continuous"
            if feature in continuous_features
            else "dynamic_categorical"
        )

    # For both training and test sets, use only same-subject information available at or before the target visit.
    train_filled[feature] = history_only_within_subject_imputation(
        train_filled, feature, ftype
    )
    test_filled[feature] = history_only_within_subject_imputation(
        test_filled, feature, ftype
    )

    # Count remaining missing values and newly imputed values.
    train_missing_after = train_filled[feature].isnull().sum()
    test_missing_after = test_filled[feature].isnull().sum()
    train_imputed = train_missing_before - train_missing_after
    test_imputed = test_missing_before - test_missing_after

    print(
        f"  {feature:<20} {train_imputed:<15} {test_imputed:<15} {train_missing_after:<17} {test_missing_after:<15}"
    )

print("\nPhase 2: Stratified temporal imputation for remaining missings...")
print("(Using actual_months_from_baseline for temporal stratification)")
print(
    "(Static features are filled consistently across all visits for the same patient)"
)
print()
print(
    f"  {'Feature':<20} {'Train imputed':<15} {'Test imputed':<15} {'Train remaining':<17} {'Test remaining':<15}"
)
print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*17} {'-'*15}")

for feature in low_missing_features:
    is_categorical = feature in categorical_features
    is_static = feature in static_features

    # Count missing values before imputation.
    train_missing_before = train_filled[feature].isnull().sum()
    test_missing_before = test_filled[feature].isnull().sum()

    # Build stratified parameters from the training data.
    params = stratified_temporal_imputation(
        train_filled, test_filled, feature, is_categorical
    )
    stratified_params[feature] = params

    # Apply to training data (static features share the same value across visits within each patient).
    train_filled[feature] = apply_stratified_imputation(
        train_filled, feature, params, is_categorical, is_static=is_static
    )

    # Apply to test data (static features share the same value across visits within each patient).
    test_filled[feature] = apply_stratified_imputation(
        test_filled, feature, params, is_categorical, is_static=is_static
    )

    # Count remaining missing values and newly imputed values.
    train_remaining = train_filled[feature].isnull().sum()
    test_remaining = test_filled[feature].isnull().sum()
    train_imputed = train_missing_before - train_remaining
    test_imputed = test_missing_before - test_remaining

    print(
        f"  {feature:<20} {train_imputed:<15} {test_imputed:<15} {train_remaining:<17} {test_remaining:<15}"
    )

print()

print("\nPhase 3: Verifying consistency for static features...")
print(
    "(Static features should have identical values across all visits for same patient)"
)
print()

for feature in static_features:
    if feature in train_filled.columns:
        # Check the training data using vectorized operations.
        train_inconsistencies = train_filled.groupby("PATNO")[feature].nunique()
        train_inconsistent_patnos = train_inconsistencies[
            train_inconsistencies > 1
        ].index

        # Check the test data using vectorized operations.
        test_inconsistencies = (
            test_filled.groupby("PATNO")[feature].nunique()
            if feature in test_filled.columns
            else pd.Series()
        )
        test_inconsistent_patnos = (
            test_inconsistencies[test_inconsistencies > 1].index
            if len(test_inconsistencies) > 0
            else []
        )

        if len(train_inconsistent_patnos) > 0 or len(test_inconsistent_patnos) > 0:
            print(
                f"  {feature:<20} WARNING: Inconsistent patients: Train={len(train_inconsistent_patnos)}, Test={len(test_inconsistent_patnos)}"
            )
            print(
                f"  {feature:<20} -> Retained; observed values are never overwritten."
            )
        else:
            print(f"  {feature:<20} Already consistent [OK]")

print()

# ============================================================================
# STEP 2: TIME-AWARE ADVANCED IMPUTATION FOR HIGH MISSING RATE FEATURES (>=10%)
# ============================================================================

print("=" * 80)
print("STEP 2: TIME-AWARE ADVANCED IMPUTATION (Missing Rate >= 10%)")
print("=" * 80)

# Separate continuous and categorical high-missing features
high_missing_continuous = [f for f in high_missing_features if f in continuous_features]
high_missing_categorical = [
    f for f in high_missing_features if f in categorical_features
]

print(f"High-missing continuous features: {len(high_missing_continuous)}")
print(f"High-missing categorical features: {len(high_missing_categorical)}")
print()

# Prepare feature set INCLUDING time-related variables
time_features_for_imputation = [
    "age_at_visit",  # Patient age at the visit.
    "duration",  # Disease duration.
    "actual_months_from_baseline",  # Exact months from baseline, accounting for irregular visit intervals.
]

predictor_features = [
    f
    for f in train_filled.columns
    if f not in exclude_features + ["months_from_baseline"]
    and f not in high_missing_features
]

# Add temporal features to the predictors.
predictor_features = list(
    dict.fromkeys(predictor_features + time_features_for_imputation)
)

print(f"Predictor features for advanced imputation: {len(predictor_features)}")
print(f"\nTime features included in prediction:")
for tf in time_features_for_imputation:
    print(f"  - {tf}")
print()

# ============================================================================
# STEP 2.1: TIME-AWARE MICE FOR CONTINUOUS FEATURES (WITH IMMEDIATE CLIPPING)
# ============================================================================

if len(high_missing_continuous) > 0:
    print("-" * 80)
    print("STEP 2.1 Time-Aware Iterative Single Imputation (via Chained Equations)")
    print(
        "(Note: Produces a deterministic single imputed dataset; not Rubin's pooled multiple imputation)"
    )
    print("-" * 80)
    print("Iterative imputer will learn relationships like:")
    print("  updrs3_score ~ age + duration + actual_months_from_baseline + ...")
    print("  (accounting for non-uniform visit intervals)")
    print()

    mice_features = predictor_features + high_missing_continuous

    train_mice = train_filled[mice_features].copy()
    test_mice = test_filled[mice_features].copy()

    # Save an imputation mask identifying imputed values.
    train_missing_mask = train_df[high_missing_continuous].isnull()
    test_missing_mask = test_df[high_missing_continuous].isnull()

    # IterativeImputer (Iterative Single Imputation based on MICE chained equations)
    mice_imputer = IterativeImputer(
        random_state=42,
        max_iter=20,  # Increase the iteration limit.
        tol=1e-3,  # Set the convergence tolerance.
        initial_strategy="median",
        imputation_order="random",  # Use random imputation order to avoid order effects.
    )

    print("Fitting iterative imputer on training data (with time features)...")
    train_mice_imputed = pd.DataFrame(
        mice_imputer.fit_transform(train_mice),
        columns=mice_features,
        index=train_mice.index,
    )

    print("Transforming test data...")
    test_mice_imputed = pd.DataFrame(
        mice_imputer.transform(test_mice), columns=mice_features, index=test_mice.index
    )

    # ===== Clip out-of-range values immediately. =====
    print("\nClipping outliers in MICE-imputed values (5th-95th percentile)...")

    # Store clipping statistics.
    clipping_stats = {}

    for feature in high_missing_continuous:
        # Calculate clipping bounds from the original training values.
        original_values = train_df[feature].dropna()
        lower_bound = original_values.quantile(0.05)
        upper_bound = original_values.quantile(0.95)

        # === Training set: Impute and clip. ===
        train_imputed_values = train_mice_imputed.loc[
            train_missing_mask[feature], feature
        ]
        train_clipped_values = train_imputed_values.clip(
            lower=lower_bound, upper=upper_bound
        )
        train_filled.loc[train_missing_mask[feature], feature] = train_clipped_values

        # Count clipped values.
        train_n_clipped = (
            (train_imputed_values < lower_bound) | (train_imputed_values > upper_bound)
        ).sum()

        # === Test set: Impute and clip. ===
        test_imputed_values = test_mice_imputed.loc[test_missing_mask[feature], feature]
        test_clipped_values = test_imputed_values.clip(
            lower=lower_bound, upper=upper_bound
        )
        test_filled.loc[test_missing_mask[feature], feature] = test_clipped_values

        # Count clipped values.
        test_n_clipped = (
            (test_imputed_values < lower_bound) | (test_imputed_values > upper_bound)
        ).sum()

        # Store statistics.
        clipping_stats[feature] = {
            "train_clipped": train_n_clipped,
            "test_clipped": test_n_clipped,
            "train_imputed": train_missing_mask[feature].sum(),
            "test_imputed": test_missing_mask[feature].sum(),
            "train_remaining": train_filled[feature].isnull().sum(),
            "test_remaining": test_filled[feature].isnull().sum(),
        }

    # Use a consistent output format.
    print()
    print(
        f"  {'Feature':<20} {'Method':<15} {'Train imputed':<15} {'Test imputed':<15} {'Train clipped':<15} {'Test clipped':<15} {'Train remaining':<17} {'Test remaining':<15}"
    )
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*17} {'-'*15}")
    for feature in high_missing_continuous:
        stats = clipping_stats[feature]
        print(
            f"  {feature:<20} {'MICE':<15} {stats['train_imputed']:<15} {stats['test_imputed']:<15} {stats['train_clipped']:<15} {stats['test_clipped']:<15} {stats['train_remaining']:<17} {stats['test_remaining']:<15}"
        )

    print()

# ============================================================================
# STEP 2.2: TIME-AWARE MODEL-BASED IMPUTATION FOR CATEGORICAL FEATURES
# ============================================================================

if len(high_missing_categorical) > 0:
    print("-" * 80)
    print("STEP 2.2 Time-Aware Model-based Imputation for Categorical Features")
    print("-" * 80)
    print("Random Forest will use time features to predict categorical outcomes")
    print("(e.g., NHY stage considering disease duration and actual months)")
    print()

    categorical_models = {}

    print(
        f"  {'Feature':<20} {'Method':<15} {'Train imputed':<15} {'Test imputed':<15} {'Train clipped':<15} {'Test clipped':<15} {'Train remaining':<17} {'Test remaining':<15}"
    )
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*17} {'-'*15}")

    for feature in high_missing_categorical:
        available_predictors = [f for f in predictor_features if f != feature]

        train_missing = train_df[feature].isnull()
        test_missing = test_df[feature].isnull()
        train_not_missing = ~train_missing

        train_imputed = train_missing.sum()
        test_imputed = test_missing.sum()

        # Use stratified imputation if the training sample is too small.
        if train_not_missing.sum() < 10:
            is_categorical = True
            params = stratified_temporal_imputation(
                train_filled, test_filled, feature, is_categorical
            )

            train_filled[feature] = apply_stratified_imputation(
                train_filled, feature, params, is_categorical
            )
            test_filled[feature] = apply_stratified_imputation(
                test_filled, feature, params, is_categorical
            )
            categorical_models[feature] = {
                "method": "stratified_mode",
                "params": params,
            }
            method_used = "Stratified"
        else:
            # Use a random forest.
            X_train = train_filled.loc[train_not_missing, available_predictors].fillna(
                0
            )
            y_train = train_filled.loc[train_not_missing, feature]

            # Adjusted random-forest parameters.
            rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=15,  # Increase the maximum depth.
                min_samples_split=10,  # Set the minimum split size.
                min_samples_leaf=5,  # Limit overfitting.
                class_weight="balanced",  # Handle class imbalance.
            )
            rf_model.fit(X_train, y_train)

            # Impute training data.
            if train_missing.sum() > 0:
                X_train_missing = train_filled.loc[
                    train_missing, available_predictors
                ].fillna(0)
                train_filled.loc[train_missing, feature] = rf_model.predict(
                    X_train_missing
                )

            # Impute test data.
            if test_missing.sum() > 0:
                X_test_missing = test_filled.loc[
                    test_missing, available_predictors
                ].fillna(0)
                test_filled.loc[test_missing, feature] = rf_model.predict(
                    X_test_missing
                )

            categorical_models[feature] = {"method": "rf", "model": rf_model}
            method_used = "Random Forest"

        train_remaining = train_filled[feature].isnull().sum()
        test_remaining = test_filled[feature].isnull().sum()

        # Categorical features do not require clipping; display N/A.
        print(
            f"  {feature:<20} {method_used:<15} {train_imputed:<15} {test_imputed:<15} {'N/A':<15} {'N/A':<15} {train_remaining:<17} {test_remaining:<15}"
        )

    print()

print("=" * 80)
print("STEP 2 COMPLETED")
print("=" * 80)
print("All high-missing features have been imputed with:")
print("  - Continuous features: MICE + immediate outlier clipping")
print("  - Categorical features: Random Forest or stratified mode")
print()

# ============================================================================
# FINAL VALIDATION AND EXPORT
# ============================================================================

print("=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

# ---- Safety assertions: Prevent leakage of the MCI outcome. ----
print("Running safety assertions...")

# Assertion 1: MCI is excluded from predictor_features.
assert "MCI" not in predictor_features, "FATAL: MCI leaked into predictor_features"

# Assertion 2: MCI is excluded from mice_features.
if len(high_missing_continuous) > 0:
    assert "MCI" not in mice_features, "FATAL: MCI leaked into mice_features"

# Assertion 3: MCI values are identical before and after imputation.
assert (
    train_filled["MCI"] == train_df["MCI"]
).all(), "FATAL: train MCI values altered during imputation"
assert (
    test_filled["MCI"] == test_df["MCI"]
).all(), "FATAL: test MCI values altered during imputation"

print("All safety assertions PASSED: MCI is not in any imputation model.")
print()

# Calculate remaining missing values (excluding time-derived features)
features_to_check = [
    col for col in train_filled.columns if col not in time_derived_features
]
train_remaining = train_filled[features_to_check].isnull().sum().sum()
test_remaining = test_filled[features_to_check].isnull().sum().sum()

print(f"Training set remaining missing values: {train_remaining}")
print(f"Test set remaining missing values: {test_remaining}")
print()

if train_remaining > 0:
    print("WARNING: Training set still has missing values:")
    missing_cols = train_filled[features_to_check].isnull().sum()
    print(missing_cols[missing_cols > 0])
    print()

if test_remaining > 0:
    print("WARNING: Test set still has missing values:")
    missing_cols = test_filled[features_to_check].isnull().sum()
    print(missing_cols[missing_cols > 0])
    print()

# Verify data integrity
print("Verifying data integrity...")
train_changed = 0
test_changed = 0

for col in train_df.columns:
    if col in exclude_features + time_derived_features:
        continue
    original_not_null = train_df[col].notna()
    if original_not_null.sum() > 0:
        changed = ~np.isclose(
            train_df.loc[original_not_null, col],
            train_filled.loc[original_not_null, col],
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )
        train_changed += changed.sum()

for col in test_df.columns:
    if col in exclude_features + time_derived_features:
        continue
    original_not_null = test_df[col].notna()
    if original_not_null.sum() > 0:
        changed = ~np.isclose(
            test_df.loc[original_not_null, col],
            test_filled.loc[original_not_null, col],
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )
        test_changed += changed.sum()

print(f"Training set: {train_changed} non-missing values changed (should be 0)")
print(f"Test set: {test_changed} non-missing values changed (should be 0)")
print()

# ============================================================================
# IMPUTATION QUALITY ASSESSMENT
# ============================================================================

print("=" * 80)
print("IMPUTATION QUALITY ASSESSMENT")
print("=" * 80)


def evaluate_imputation_quality(df_original, df_filled, feature, dataset_name=""):
    """
    Evaluate imputation quality.

    Checks:
    1. Distribution consistency (KS test).
    2. Range checks (values outside the original range).
    3. Comparison of basic statistics.
    """
    imputed_mask = df_original[feature].isna()
    n_imputed = imputed_mask.sum()

    if n_imputed == 0:
        return None

    imputed_values = df_filled.loc[imputed_mask, feature]
    original_values = df_original[feature].dropna()

    result = {
        "feature": feature,
        "dataset": dataset_name,
        "n_imputed": n_imputed,
        "n_original": len(original_values),
    }

    # Apply KS tests and range checks to continuous features only.
    if (
        feature in continuous_features
        and len(original_values) > 0
        and len(imputed_values) > 0
    ):
        # 1. KS test: Compare distributions of imputed and original values.
        try:
            ks_stat, p_value = ks_2samp(imputed_values, original_values)
            result["ks_stat"] = ks_stat
            result["ks_pvalue"] = p_value
        except:
            result["ks_stat"] = None
            result["ks_pvalue"] = None

        # 2. Range check: Identify imputed values outside the original range.
        original_min, original_max = original_values.min(), original_values.max()
        outliers = (
            (imputed_values < original_min) | (imputed_values > original_max)
        ).sum()
        result["outliers"] = outliers
        result["outlier_rate"] = outliers / n_imputed * 100

        # 3. Basic statistics.
        result["original_mean"] = original_values.mean()
        result["imputed_mean"] = imputed_values.mean()
        result["original_std"] = original_values.std()
        result["imputed_std"] = imputed_values.std()
    else:
        result["ks_stat"] = None
        result["ks_pvalue"] = None
        result["outliers"] = None
        result["outlier_rate"] = None
        result["original_mean"] = None
        result["imputed_mean"] = None
        result["original_std"] = None
        result["imputed_std"] = None

    return result


# Evaluate all imputed features.
all_imputed_features = low_missing_features + high_missing_features
quality_results = []

# Collect quality assessments for the training set.
for feature in all_imputed_features:
    if feature in train_df.columns and feature in train_filled.columns:
        result = evaluate_imputation_quality(train_df, train_filled, feature, "Train")
        if result:
            quality_results.append(result)

# Collect quality assessments for the test set.
for feature in all_imputed_features:
    if feature in test_df.columns and feature in test_filled.columns:
        result = evaluate_imputation_quality(test_df, test_filled, feature, "Test")
        if result:
            quality_results.append(result)

# Create a quality-assessment DataFrame.
quality_df = pd.DataFrame(quality_results)

# Print the training-set table.
print("\n[Training Set Quality Assessment]")
print()
train_quality = quality_df[quality_df["dataset"] == "Train"].copy()

if len(train_quality) > 0:
    print(
        f"  {'Feature':<12} {'N_Imputed':<10} {'N_Original':<11} {'KS_Stat':<8} {'KS_P-value':<11} {'Outliers':<9} {'Outlier_%':<10} {'Orig_Mean':<10} {'Imp_Mean':<9}"
    )
    print(
        f"  {'-'*12} {'-'*10} {'-'*11} {'-'*8} {'-'*11} {'-'*9} {'-'*10} {'-'*10} {'-'*9}"
    )

    for _, row in train_quality.iterrows():
        feature = row["feature"]
        n_imputed = int(row["n_imputed"])
        n_original = int(row["n_original"])
        ks_stat = f"{row['ks_stat']:.4f}" if pd.notna(row["ks_stat"]) else "N/A"
        ks_pvalue = f"{row['ks_pvalue']:.4f}" if pd.notna(row["ks_pvalue"]) else "N/A"
        outliers = f"{int(row['outliers'])}" if pd.notna(row["outliers"]) else "N/A"
        outlier_rate = (
            f"{row['outlier_rate']:.1f}%" if pd.notna(row["outlier_rate"]) else "N/A"
        )
        orig_mean = (
            f"{row['original_mean']:.2f}" if pd.notna(row["original_mean"]) else "N/A"
        )
        imp_mean = (
            f"{row['imputed_mean']:.2f}" if pd.notna(row["imputed_mean"]) else "N/A"
        )

        print(
            f"  {feature:<12} {n_imputed:<10} {n_original:<11} {ks_stat:<8} {ks_pvalue:<11} {outliers:<9} {outlier_rate:<10} {orig_mean:<10} {imp_mean:<9}"
        )

# Print the test-set table.
print("\n\n[Test Set Quality Assessment]")
print()
test_quality = quality_df[quality_df["dataset"] == "Test"].copy()

if len(test_quality) > 0:
    print(
        f"  {'Feature':<12} {'N_Imputed':<10} {'N_Original':<11} {'KS_Stat':<8} {'KS_P-value':<11} {'Outliers':<9} {'Outlier_%':<10} {'Orig_Mean':<10} {'Imp_Mean':<9}"
    )
    print(
        f"  {'-'*12} {'-'*10} {'-'*11} {'-'*8} {'-'*11} {'-'*9} {'-'*10} {'-'*10} {'-'*9}"
    )

    for _, row in test_quality.iterrows():
        feature = row["feature"]
        n_imputed = int(row["n_imputed"])
        n_original = int(row["n_original"])
        ks_stat = f"{row['ks_stat']:.4f}" if pd.notna(row["ks_stat"]) else "N/A"
        ks_pvalue = f"{row['ks_pvalue']:.4f}" if pd.notna(row["ks_pvalue"]) else "N/A"
        outliers = f"{int(row['outliers'])}" if pd.notna(row["outliers"]) else "N/A"
        outlier_rate = (
            f"{row['outlier_rate']:.1f}%" if pd.notna(row["outlier_rate"]) else "N/A"
        )
        orig_mean = (
            f"{row['original_mean']:.2f}" if pd.notna(row["original_mean"]) else "N/A"
        )
        imp_mean = (
            f"{row['imputed_mean']:.2f}" if pd.notna(row["imputed_mean"]) else "N/A"
        )

        print(
            f"  {feature:<12} {n_imputed:<10} {n_original:<11} {ks_stat:<8} {ks_pvalue:<11} {outliers:<9} {outlier_rate:<10} {orig_mean:<10} {imp_mean:<9}"
        )

# Print summary statistics.
print("\n\n[Summary Statistics]")
print("-" * 60)
continuous_results = quality_df[quality_df["ks_pvalue"].notna()]
if len(continuous_results) > 0:
    ks_pass = (continuous_results["ks_pvalue"] > 0.05).sum()
    ks_total = len(continuous_results)
    print(
        f"KS test passed (p > 0.05): {ks_pass}/{ks_total} ({ks_pass/ks_total*100:.1f}%)"
    )

    low_outlier = (continuous_results["outlier_rate"] < 5).sum()
    print(
        f"Low outlier rate (< 5%): {low_outlier}/{ks_total} ({low_outlier/ks_total*100:.1f}%)"
    )

print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Remove temporary columns before saving
train_filled = train_filled.drop(columns=time_derived_features, errors="ignore")
test_filled = test_filled.drop(columns=time_derived_features, errors="ignore")

print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Verify no unexpected NaNs remain in feature columns
train_feat_cols = [c for c in train_filled.columns if c not in exclude_features]
test_feat_cols = [c for c in test_filled.columns if c not in exclude_features]
train_nans = train_filled[train_feat_cols].isna().sum().sum()
test_nans = test_filled[test_feat_cols].isna().sum().sum()
print(f"Remaining NaNs in feature columns -> Train: {train_nans}, Test: {test_nans}")
if train_nans > 0 or test_nans > 0:
    print("WARNING: Some features still contain missing values!")

train_output = "PPMI_7_train.csv"
test_output = "PPMI_7_test.csv"

train_filled.to_csv(train_output, index=False)
test_filled.to_csv(test_output, index=False)

print(f"Training set saved to: {train_output}")
print(f"Test set saved to: {test_output}")
print()
