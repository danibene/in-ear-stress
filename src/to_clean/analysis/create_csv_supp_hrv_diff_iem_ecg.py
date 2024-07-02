from pathlib import Path

import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from stresspred import (
    AudaceDataLoader,
    get_cv_iterator,
    make_prediction_pipeline,
    code_paths,
)

feat_in_table = pd.read_csv(Path(code_paths["repo_path"], "local_data/neurokit2_hrv_feat_info.csv"))
feats_in_paper = set(feat_in_table["Feature"])
feature_list_path = Path(code_paths["repo_path"],"expected_features.json")
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
pipe_clf = make_prediction_pipeline(est=LogisticRegression(random_state=seed, max_iter=1000))
base_cond = "Without baselining"
out = AudaceDataLoader().get_split_pred_df(
                selected_tasks=selected_tasks,
                selected_signals=selected_signals,
                load_from_file=True,
                save_file=False,
                rel_values= base_cond=="With baselining",
            )
X = out["X"]
y = out["y"]
sub = out["sub"]
task = out["task"]
signal = out["signal"]
nk_feats = [col for col in out["X"].columns if "HRV_" in col]
sig_name_train = "IEML"
sig_name_val = "IEML"
sig_names_train = sig_name_train.split(" + ")
outer_cv, _ = get_cv_iterator(
    sub,
    n_outer_splits=5,
    n_inner_splits=4,
    train_bool=np.array([True if sig in sig_names_train else False for sig in signal]),
    val_bool=signal == sig_name_val
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
    coefs.append(dict(zip(selected_features, list(pipe_clf.named_steps["estimator"].coef_.flatten()))))


print(np.mean(scores))
coef_df = pd.DataFrame(coefs)
coef_df.index.name = "Feature"
coef_df = pd.melt(coef_df, var_name="Feature", value_name="Coefficient")
coef_df["Signal train"] = sig_name_train
coef_df["Signal val"] = sig_name_val
coef_df["Baselining"] = base_cond
coef_dfs.append(coef_df)

coef_df = pd.concat(coef_dfs)
coef_df["Abs Coef"] = np.abs(coef_df["Coefficient"])
# Create table of mean and std of coefficients for each feature
# First compute mean absolute coefficient
summary = coef_df.groupby(["Feature"])["Abs Coef"].mean().reset_index().sort_values(by="Abs Coef", ascending=False).reset_index()
# Then add standard deviation
std = coef_df.groupby(["Feature"])["Abs Coef"].std().reset_index()
summary = summary.merge(std, on="Feature", suffixes=(" mean", " std"))

categories = []
for feature in summary["Feature"].values:
    category = feat_in_table[feat_in_table["Feature"] == feature.split("HRV_")[1]]["Category"].values[0]
    categories.append(category)

summary["Category"] = categories
output_path = Path(code_paths["repo_path"], "local_data", "feature_importance_lr_train_iem.csv")
summary[["Feature", "Abs Coef mean", "Abs Coef std", "Category"]].to_csv(output_path, index=True)
