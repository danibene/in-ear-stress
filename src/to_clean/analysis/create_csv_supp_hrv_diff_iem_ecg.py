from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import json
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from stresspred import (
    AudaceDataLoader,
    get_cv_iterator,
    make_prediction_pipeline,
    code_paths,
)

root_out_path = Path(code_paths["repo_path"], "local_data/outputs")
feat_in_table = pd.read_csv(
    Path(code_paths["repo_path"], "local_data/neurokit2_hrv_feat_info.csv")
)
feats_in_paper = set(feat_in_table["Feature"])
feature_list_path = Path(code_paths["repo_path"], "expected_features.json")
with open(feature_list_path, "r") as json_file:
    expected_features = json.load(json_file)
feats_in_model = set([f.split("HRV_")[1] for f in expected_features])
shared_feats = feats_in_model.intersection(feats_in_paper)
expected_columns = ["HRV_" + f for f in shared_feats]

coef_dfs = []
selected_tasks = ["MENTAL", "CPT"]
selected_signals = ["ECG", "IEML", "ECG-IEMLe"]
list_sig_name_train = [
    "ECG-IEMLe + ECG",
    "IEML + ECG",
    "ECG",
    "IEML",
    "ECG-IEMLe",
]
list_sig_name_val = ["ECG", "IEML", "ECG-IEMLe"]

seed = 0
pipe_clf = make_prediction_pipeline(
    est=LogisticRegression(random_state=seed, max_iter=1000)
)
base_cond = "Without baselining"

out = AudaceDataLoader().get_split_pred_df(
    selected_tasks=selected_tasks,
    selected_signals=selected_signals,
    load_from_file=True,
    save_file=False,
    rel_values=base_cond == "With baselining",
)
X = out["X"]
y = out["y"]
sub = out["sub"]
task = out["task"]
signal = out["signal"]
nk_feats = [col for col in out["X"].columns if "HRV_" in col]
sig_names_train = ["ECG", "IEML"]
for sig_name_train in sig_names_train:
    # sig_name_train = "IEML"
    sig_name_val = "IEML"
    sig_names_train = sig_name_train.split(" + ")
    outer_cv, _ = get_cv_iterator(
        sub,
        n_outer_splits=5,
        n_inner_splits=4,
        train_bool=np.array(
            [True if sig in sig_names_train else False for sig in signal]
        ),
        val_bool=signal == sig_name_val,
    )
    selected_features = expected_columns
    coefs = []
    scores = []
    for train, test in outer_cv:
        X_train = X.iloc[train].loc[:, selected_features]
        y_train = y[train]
        pipe_clf.fit(X_train, y_train)
        X_test = X.iloc[test].loc[:, selected_features]
        y_test = y[test]
        scores.append(pipe_clf.score(X_test, y_test))
        coef_dict = dict(
            zip(
                selected_features,
                list(pipe_clf.named_steps["estimator"].coef_.flatten()),
            )
        )
        coefs.append(coef_dict)

    print(np.mean(scores))
    coef_df = pd.DataFrame(coefs)
    coef_df.index.name = "Feature"
    coef_df = pd.melt(coef_df, var_name="Feature", value_name="Coefficient")
    coef_df["Signal train"] = sig_name_train
    coef_df["Signal val"] = sig_name_val
    coef_df["Baselining"] = base_cond
    coef_dfs.append(coef_df)

coef_df = pd.concat(coef_dfs)

iem_and_ecg_coef_df = coef_df.copy(deep=False)
coef_df = coef_df[coef_df["Signal train"] == "IEML"]

coef_df["Abs Coef"] = np.abs(coef_df["Coefficient"])
# Create table of mean and std of coefficients for each feature
# First compute mean absolute coefficient
summary = (
    coef_df.groupby(["Feature"])["Abs Coef"]
    .mean()
    .reset_index()
    .sort_values(by="Abs Coef", ascending=False)
    .reset_index()
)
order = summary.sort_values(by="Abs Coef", ascending=False)["Feature"].values
# Then add standard deviation
std = coef_df.groupby(["Feature"])["Abs Coef"].std().reset_index()
summary = summary.merge(std, on="Feature", suffixes=(" mean", " std"))

# Plot coefficients for all features
fig, ax = plt.subplots(figsize=(15, 8))
# Create boxplot
sns.boxplot(
    x="Feature",
    y="Coefficient",
    hue="Signal train",
    data=iem_and_ecg_coef_df,
    ax=ax,
    order=order,
    palette="Set3",
)
# Rotate x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# Remove "HRV_" from x-axis labels
ax.set_xticklabels(
    [str(x).split("HRV_")[1].split("'")[0] for x in ax.get_xticklabels()]
)
# Add title
# ax.set_title("Coefficients for all features")
plt.tight_layout()
plt.show()


# Then add the 75th percentile coefficient
summary["Coef 75th percentile"] = (
    coef_df.groupby(["Feature"])["Coefficient"]
    .quantile(0.75)
    .reset_index()["Coefficient"]
)
# Then add the 25th percentile coefficient
summary["Coef 25th percentile"] = (
    coef_df.groupby(["Feature"])["Coefficient"]
    .quantile(0.25)
    .reset_index()["Coefficient"]
)

categories = []
for feature in summary["Feature"].values:
    category = feat_in_table[feat_in_table["Feature"] == feature.split("HRV_")[1]][
        "Category"
    ].values[0]
    categories.append(category)

summary["Category"] = categories
output_path = Path(
    code_paths["repo_path"], "local_data", "feature_importance_lr_train_iem.csv"
)
summary[
    [
        "Feature",
        "Abs Coef mean",
        "Abs Coef std",
        "Category",
        "Coef 75th percentile",
        "Coef 25th percentile",
    ]
].to_csv(output_path, index=True)
# select top features by mean absolute coefficient
top_features = summary["Feature"].values[:4]

iem_and_ecg_coef_df = iem_and_ecg_coef_df[
    iem_and_ecg_coef_df["Feature"].isin(top_features)
]

# Plot coefficients for top 4 features
fig, ax = plt.subplots(figsize=(5, 4))
# Create boxplot
sns.boxplot(
    x="Feature",
    y="Coefficient",
    hue="Signal train",
    data=iem_and_ecg_coef_df,
    ax=ax,
    order=top_features,
    palette="Set3",
)
# Rotate x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# Remove "HRV_" from x-axis labels
ax.set_xticklabels(
    [str(x).split("HRV_")[1].split("'")[0] for x in ax.get_xticklabels()]
)
plt.tight_layout()
plt.show()

abs_df, _ = AudaceDataLoader().get_feat_dfs(save_file=True)
# abs_df = pd.read_csv("local_data/abs_feat_df_20230411.csv")
df_mental_cpt = abs_df[abs_df["Task"].isin(["MENTAL", "CPT"])]
gt_record = df_mental_cpt[df_mental_cpt["Signal"] == "ECG"]
record = df_mental_cpt[df_mental_cpt["Signal"] == "IEML"]

# top_features = ["HRV_MedianNN", "HRV_RMSSD"]
for hrv_col in top_features:

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=True,
        sharey=True,
        squeeze=True,
        figsize=(8, 4),
        dpi=1000,
    )

    dfs_list = []

    conditions = []
    for cond in np.unique(gt_record["Rest"]):
        cond_ind = np.where(np.array(record["Rest"]) == cond)[0]
        cond_ind_gt = np.where(np.array(gt_record["Rest"]) == cond)[0]
        df = pd.DataFrame(
            {
                "ECG": np.array(gt_record[hrv_col])[cond_ind_gt],
                "IEM": np.array(record[hrv_col])[cond_ind],
            }
        )
        dfs_list.append(df)
        conditions.append(cond)
    colors = {"Rest": "grey", "Stress": "grey"}
    markers = {"Rest": "o", "Stress": "o"}

    for i in range(len(dfs_list)):

        df = dfs_list[i]
        data1 = df["ECG"]
        data2 = df["IEM"]
        sm.graphics.mean_diff_plot(
            data1, data2, ax=axs[i], scatter_kwds={"color": "#F79646", "alpha": 0.25}
        )

        axs[i].set(
            xlabel="Mean of ECG and IEM",
            ylabel="Difference between ECG and IEM",
        )
        axs[i].set_title(label=conditions[i] + " conditions", fontsize=20)

    # Increase x-axis and y-axis limits by adding 10% of the range to the left and right
    PERCENTAGE = 0.1
    for ax in axs:
        ax.set_xlim(
            [
                ax.get_xlim()[0] - PERCENTAGE * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                ax.get_xlim()[1] + PERCENTAGE * (ax.get_xlim()[1] - ax.get_xlim()[0]),
            ]
        )
        ax.set_ylim(
            [
                ax.get_ylim()[0] - PERCENTAGE * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                ax.get_ylim()[1] + PERCENTAGE * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            ]
        )

    only_feature_name = hrv_col.split("HRV_")[1]
    plt.suptitle(only_feature_name, y=1.05, fontsize=20)

    fig_name = "bland-altman_" + only_feature_name + "_IEM.png"
    fig_path = Path(root_out_path, fig_name)
    fig.savefig(fig_path, facecolor="white", transparent=False, bbox_inches="tight")
    plt.show()
