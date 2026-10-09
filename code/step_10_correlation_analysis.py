import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import figstyle as fs

fs.apply()
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"

# Read the main dataset.
df = pd.read_csv("PPMI_7_train.csv")

# Candidate features for correlation screening; only a subset is displayed, and these are not all retained in the final model.
features = [
    "age_at_visit",
    "SEX",
    "EDUCYRS",
    "fampd_bin",
    "BMI",
    "agediag",
    "ageonset",
    "sym_tremor",
    "sym_rigid",
    "sym_brady",
    "sym_posins",
    "sym_other",
    "PDTRTMNT",
    "LEDD",
    "quip",
    "ess",
    "rem",
    "gds",
    "stai_state",
    "stai_trait",
    "scopa",
    "orthostasis",
    "NHY",
    "pigd",
    "td_pigd",
    "updrs2_score",
    "updrs3_score",
    "APOE_e4",
    "has_GBA",
    "has_LRRK2",
    "duration",
]

# Read the feature-to-abbreviation mapping.
_csv_df = pd.read_csv("PPMI_feature_mapping.csv")
feature_name_mapping = dict(zip(_csv_df["feature"], _csv_df["abbreviation"]))

# Select the original values of these features.
df_features = df[features]

# Calculate the Spearman rank-correlation matrix.
# The candidate pool includes binary variables (SEX, fampd_bin, sym_*), ordinal NHY, and non-normal continuous variables.
# Spearman rank correlation is more appropriate for these mixed distributions than Pearson correlation.
correlation_matrix = df_features.corr(method="spearman")

# Set the correlation threshold for display and analysis.
CORRELATION_THRESHOLD = 0.6


# Select features with an absolute correlation at or above the threshold with any other feature.
# This selection is for analysis and heatmap display; it does not retain all features in the final model.
def get_features_above_correlation_threshold(corr_matrix, threshold=0.2):
    """
    Select features exceeding the absolute-correlation threshold with any other feature for analysis/heatmap display.
    """
    features_selected = []
    features_unselected = []
    for i, feature in enumerate(corr_matrix.columns):
        # Exclude self-correlation and inspect correlations with all other features.
        correlations = corr_matrix.iloc[i, :].drop(feature)
        if (abs(correlations) >= threshold).any():
            features_selected.append(feature)
        else:
            features_unselected.append(feature)
    return features_selected, features_unselected


# Select the features for the heatmap.
features_for_heatmap, features_with_consistently_low_corr = (
    get_features_above_correlation_threshold(
        correlation_matrix, threshold=CORRELATION_THRESHOLD
    )
)

print("=" * 80)
print(
    f"可用于相关性热图展示的特征 (与任意其他特征的相关系数绝对值 >= {CORRELATION_THRESHOLD}):"
)
print("=" * 80)
print(f"\n可纳入展示的特征数量: {len(features_for_heatmap)}")
print(f"始终低相关且未纳入的特征数量: {len(features_with_consistently_low_corr)}")

if features_with_consistently_low_corr:
    print(
        f"\n以下特征与所有其他特征的相关系数绝对值均 < {CORRELATION_THRESHOLD}（未纳入相关性热图）:"
    )
    for feature in features_with_consistently_low_corr:
        max_corr = correlation_matrix[feature].drop(feature).abs().max()
        print(f"  - {feature:20s} (最大相关系数: {max_corr:.4f})")
else:
    print(f"\n所有特征都至少与一个其他特征有高于阈值的相关性")

print("\n可用于热力图展示的特征（并非最终用于模型的全部特征）:")
for feature in features_for_heatmap:
    max_corr = correlation_matrix[feature].drop(feature).abs().max()
    print(f"  - {feature:20s} (最大相关系数: {max_corr:.4f})")

# Plot the correlated feature subset to visualize redundancy; further feature removal is performed downstream.
correlation_matrix_filtered = correlation_matrix.loc[
    features_for_heatmap, features_for_heatmap
]

# Use abbreviations for row and column labels.
correlation_matrix_filtered_renamed = correlation_matrix_filtered.rename(
    index=feature_name_mapping, columns=feature_name_mapping
)

# Mask the upper triangle to display only the lower triangle.
mask = np.triu(np.ones_like(correlation_matrix_filtered_renamed, dtype=bool))

# Stretch RdBu_r colors at correlations +/-0.75 to +/-1 while retaining the near-white center.
from matplotlib.colors import LinearSegmentedColormap

HM_CMAP = LinearSegmentedColormap.from_list(
    "RdBu_r_middle75", plt.get_cmap("RdBu_r")(np.linspace(0.125, 0.875, 256))
)

# Use a square heatmap with square cells, enlarged for the 11x11 numeric annotations.
HM_W = 88.0
HM_H = HM_W
fig, axes_grid = fs.axes_grid(
    1, 1, ax_w=HM_W, ax_h=HM_H, left=20.0, right=14.0, bottom=14.0
)
ax = axes_grid[0, 0]
cax = fig.add_axes(
    [
        (20.0 + HM_W + 3.0) / (20.0 + HM_W + 14.0),
        14.0 / (7.0 + HM_H + 14.0),
        2.5 / (20.0 + HM_W + 14.0),
        HM_H * (10 / 11) / (7.0 + HM_H + 14.0),
    ]
)  # The first row is blank, so the color bar is one cell shorter.
sns.heatmap(
    correlation_matrix_filtered_renamed,
    mask=mask,
    annot=True,
    fmt=".2f",
    annot_kws={"fontsize": fs.LEGEND_SIZE},
    cmap=HM_CMAP,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    ax=ax,
    cbar_ax=cax,
)
cax.tick_params(labelsize=fs.LEGEND_SIZE)
ax.tick_params(axis="x", rotation=45)
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
ax.tick_params(axis="y", rotation=0)
fig.savefig("step_10_correlation_heatmap.pdf", bbox_inches="tight")
