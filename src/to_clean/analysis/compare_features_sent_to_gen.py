import numpy as np
import pandas as pd

from stresspred import AudaceDataLoader

# Load generated features
gen_abs_df, _ = AudaceDataLoader().get_feat_dfs(save_file=True)

# Load sent features
sent_abs_df = pd.read_csv("local_data/abs_feat_df_202405.csv")

# Load original features
orig_abs_df = pd.read_csv("local_data/abs_feat_df_20230411.csv")

feature = "HRV_RMSSD"
for abs_df in [gen_abs_df, sent_abs_df, orig_abs_df]:
    print(abs_df.head())
    gt_record = abs_df[abs_df["Signal"]=="ECG"]
    record = abs_df[abs_df["Signal"]=="IEML"]
    mean_abs_diff = np.nanmean(np.abs(gt_record[feature].values - record[feature].values))
    print(feature, mean_abs_diff)


# Check if features are the same
gen_abs_df_aligned = gen_abs_df.reindex_like(orig_abs_df)
print(gen_abs_df_aligned == orig_abs_df)

# For each feature, print the mean absolute difference between original and generated
# features
for feature in gen_abs_df_aligned.columns:
    # If the feature is numeric
    if gen_abs_df_aligned[feature].dtype == "float64":
        mean_abs_diff = np.nanmean(np.abs(gen_abs_df_aligned[feature] - orig_abs_df[feature]))
        # print(feature, mean_abs_diff)
