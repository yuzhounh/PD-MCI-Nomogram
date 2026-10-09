import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_subject_proxy_labels(y, subjects):
    """Create subject-level proxy labels (subject is 1 if any sample is 1)"""
    y = np.asarray(y)
    subjects = np.asarray(subjects)

    # Subject label mapping: 1 if any sample is 1
    subject_labels_map = pd.Series(y, index=subjects).groupby(level=0).max()

    # Assign each sample its subject's proxy label
    y_proxy = pd.Series(subjects).map(subject_labels_map).values

    return y_proxy, subject_labels_map.to_dict()


def print_split_statistics(train_data, test_data, subject_col="PATNO", label_col="MCI"):
    """Print data split statistics"""
    print(f"Training set shape: {train_data.shape}")
    print(f"Test set shape: {test_data.shape}")

    # Sample-level statistics
    print(f"\nSample-level label distribution:")
    print(
        f"Training set label distribution: {train_data[label_col].value_counts().to_dict()}"
    )
    print(
        f"Test set label distribution: {test_data[label_col].value_counts().to_dict()}"
    )

    train_class_1_ratio = train_data[label_col].value_counts(normalize=True).get(1, 0)
    test_class_1_ratio = test_data[label_col].value_counts(normalize=True).get(1, 0)
    difference = train_class_1_ratio - test_class_1_ratio

    print(f"Proportion of class 1 in training set: {train_class_1_ratio:.4f}")
    print(f"Proportion of class 1 in test set: {test_class_1_ratio:.4f}")
    print(f"Proportion difference: {difference:.4f}")

    # Subject-level statistics
    train_subjects = train_data[subject_col].unique()
    test_subjects = test_data[subject_col].unique()

    print(f"\nNumber of subjects in training set: {len(train_subjects)}")
    print(f"Number of subjects in test set: {len(test_subjects)}")

    # Check for overlap
    overlapping_subjects = set(train_subjects) & set(test_subjects)
    if overlapping_subjects:
        raise RuntimeError(
            f"Training and test sets have overlapping subjects: {overlapping_subjects}"
        )
    else:
        print("No subject overlap between training and test sets.")

    # Subject-level label distribution
    _, train_subject_labels_map = create_subject_proxy_labels(
        train_data[label_col].values, train_data[subject_col].values
    )
    _, test_subject_labels_map = create_subject_proxy_labels(
        test_data[label_col].values, test_data[subject_col].values
    )

    train_subject_labels = pd.Series(list(train_subject_labels_map.values()))
    test_subject_labels = pd.Series(list(test_subject_labels_map.values()))

    print(f"\nSubject-level label distribution (any sample is 1 rule):")
    print(
        f"Training set subject label distribution: {train_subject_labels.value_counts().to_dict()}"
    )
    print(
        f"Test set subject label distribution: {test_subject_labels.value_counts().to_dict()}"
    )

    train_subject_class_1_ratio = train_subject_labels.value_counts(normalize=True).get(
        1, 0
    )
    test_subject_class_1_ratio = test_subject_labels.value_counts(normalize=True).get(
        1, 0
    )
    subject_difference = train_subject_class_1_ratio - test_subject_class_1_ratio

    print(
        f"Proportion of class 1 subjects in training set: {train_subject_class_1_ratio:.4f}"
    )
    print(
        f"Proportion of class 1 subjects in test set: {test_subject_class_1_ratio:.4f}"
    )
    print(f"Subject-level proportion difference: {subject_difference:.4f}")


def stratified_group_split(
    data, test_size=0.3, random_state=42, subject_col="PATNO", label_col="MCI"
):
    """Split data using subject-level stratified sampling, ensuring data from the same subject doesn't span sets"""
    # Create proxy labels
    y_proxy, subject_labels_map = create_subject_proxy_labels(
        data[label_col].values, data[subject_col].values
    )

    # Get unique subjects and their corresponding proxy labels
    unique_subjects = np.array(list(subject_labels_map.keys()))
    subject_proxy_labels = np.array(list(subject_labels_map.values()))

    # Perform stratified split at subject level
    train_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        stratify=subject_proxy_labels,
        random_state=random_state,
    )

    # Split data based on subjects
    train_data = data[data[subject_col].isin(train_subjects)]
    test_data = data[data[subject_col].isin(test_subjects)]

    # Print statistics
    print_split_statistics(train_data, test_data, subject_col, label_col)

    return train_data, test_data


def main():
    """Main function"""
    print("=== Dataset Split ===\n")

    # Read data
    input_file = "PPMI_4_features.csv"
    print(f"Reading file: {input_file}")
    data = pd.read_csv(input_file)
    print(f"Data shape: {data.shape}")
    print(f"Column names: {list(data.columns)}\n")

    # Perform subject-level stratified split
    print("=== Performing subject-level stratified split ===\n")
    train_data, test_data = stratified_group_split(
        data, test_size=0.3, random_state=42, subject_col="PATNO", label_col="MCI"
    )

    split_counts = (
        len(train_data),
        train_data["PATNO"].nunique(),
        len(test_data),
        test_data["PATNO"].nunique(),
    )
    if split_counts != (5619, 1171, 2198, 502):
        raise RuntimeError(
            "Locked subject-level split changed: expected "
            "5619 visits/1171 subjects in training and 2198 visits/502 subjects "
            f"in validation, got {split_counts}"
        )

    # Save results
    train_output = "PPMI_5_train.csv"
    test_output = "PPMI_5_test.csv"

    train_data.to_csv(train_output, index=False)
    test_data.to_csv(test_output, index=False)

    print(f"\n=== Save Complete ===")
    print(f"Training set saved to: {train_output}")
    print(f"Test set saved to: {test_output}")


if __name__ == "__main__":
    main()
