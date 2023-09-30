import os
import pathlib
import sys
import numpy as np
import argparse
from warnings import warn
import pandas as pd

parser = argparse.ArgumentParser(
    description="HRV feature extraction with shorter segments"
)
parser.add_argument("--root_out_path", default=None, help="where to save outputs")
parser.add_argument(
    "--root_repo", default=None, help="root of the p5-stress-classifier repository"
)
parser.add_argument(
    "--root_AudaceData", default=None, help="directory containing AUDACE data"
)
parser.add_argument(
    "--root_P5_StressData", default=None, help="directory containing P5_Stress data"
)

if __name__ == "__main__":
    args = parser.parse_args()
    root_out_path = args.root_out_path
    root_repo = args.root_repo
    root_AudaceData = args.root_AudaceData
    root_P5_StressData = args.root_P5_StressData

    if root_out_path is None:
        root_out_path = r"Z:\Shared\Documents\RD\RD2\_Projets\Pascal\P5 In-Ear Biosignal Monitoring\Bridge\DanielleEersBridge\results\_gitSave\shorter_segments"
    # load the code in this repository as a package
    code_paths = {}
    code_paths["repo_name"] = "p5-stress-classifier"

    # if the location of this repository is not explicitly given
    # try to locate it by going through parent directories
    if root_repo is None:
        code_paths["repo_path"] = os.getcwd()
        base_dir = os.path.basename(code_paths["repo_path"])
        count_move_up = 0
        while base_dir != code_paths["repo_name"]:
            code_paths["repo_path"] = os.path.dirname(
                os.path.abspath(code_paths["repo_path"])
            )
            base_dir = os.path.basename(code_paths["repo_path"])
            count_move_up += 1
            if count_move_up > 10:
                warn(
                    "Root of "
                    + code_paths["repo_name"]
                    + "repository could not be automatically found. "
                    + "Please explicitly provide the repository path "
                    + "with the root_repo argument."
                )
    else:
        code_paths["repo_path"] = root_repo
    package_dir = pathlib.Path(code_paths["repo_path"], "src")
    sys.path.append(str(package_dir))

    # imports from this repository
    from stresspred import (
        code_paths,
        AudaceDataLoader,
    )

    # the acceptable percentage of missing data calculated by comparing the
    # sum of the interbeat intervals to the specified duration
    # segments with too much missing data are dropped
    sum_ibi_tol_perc = 25

    # the acceptable difference between the actual segment duration and the
    # specified segment duration, in seconds
    # (since interbeat intervals are not uniformly sampled need some tolerance)
    max_minus_min_ibi_tol = 5

    # the hop length, i.e. after how many seconds a new
    # segment of the interbeat intervals begins
    hop_len = 15

    # names of columns in interbeat interval spreadsheet
    sub_seg_key = "SubSegIdx"  # full 180s segments will have just one sub-segment
    rri_key = "Ibi"
    rri_time_key = "IbiTime"

    # where features from shorter segments should be saved
    all_derivatives_path = pathlib.Path(root_out_path, "all_derivatives")

    corrs = {}
    feat_sets = {}

    for seg_len in np.arange(1, 13) * 15:

        sum_ibi_tol = seg_len * (sum_ibi_tol_perc / 100)
        loader = AudaceDataLoader()
        abs_df_orig, _ = loader.get_feat_dfs(load_from_file=True, save_file=False)

        ibi_df = loader.get_ibi_df(load_from_file=True, save_file=False)
        seg_ibi_df = loader.resegment_ibi_df(
            seg_len=seg_len,
            hop_len=hop_len,
            in_data=ibi_df.copy(),
            sum_ibi_tol=sum_ibi_tol,
            max_minus_min_ibi_tol=max_minus_min_ibi_tol,
        )
        train_loader_derivatives_path = pathlib.Path(
            all_derivatives_path,
            loader.dataset_label,
            "seg_len" + str(seg_len),
            "hop_len" + str(hop_len),
        )
        loader.set_paths(
            data_derivatives_dir=pathlib.Path(train_loader_derivatives_path)
        )
        loader.get_ibi_df(load_from_file=False, save_file=True, in_data=seg_ibi_df)
        abs_df_seg, _ = loader.get_feat_dfs(load_from_file=True, save_file=True)

        # only use ECG data
        sel_abs_df_seg = abs_df_seg[(abs_df_seg["Signal"] == "ECG")]
        sel_abs_df_orig = abs_df_orig[(abs_df_orig["Signal"] == "ECG")]

        info_keys = [
            col for col in ibi_df.columns if col not in [rri_key, rri_time_key]
        ]

        # match by participant, task, rest
        match_index_keys = [k for k in info_keys if k != sub_seg_key]

        # take the correlation of original 180s segments with
        # the mean of the different sub-segments
        corr_mean_seg_orig = (
            sel_abs_df_seg.groupby(match_index_keys)
            .mean()
            .reset_index(drop=True)
            .corrwith(
                sel_abs_df_orig.groupby(match_index_keys).mean().reset_index(drop=True),
                method="spearman",
            )
        )

        for n_feats in [5, 10, 25, 50]:
            feat_sets["top_" + str(n_feats) + "_nk_feats_" + "seg" + str(seg_len)] = [
                feat
                for feat in corr_mean_seg_orig.sort_values(ascending=False).index
                if "HRV_" in feat
            ][:n_feats]
        corrs[seg_len] = corr_mean_seg_orig.to_dict()
    corr_df = pd.DataFrame.from_dict(corrs)
    corr_df = corr_df.drop(index=[sub_seg_key])
    corr_df.to_csv(pathlib.Path(root_out_path, "corr_hrv_short_segs.csv"))