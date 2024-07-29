import numpy as np
import pandas as pd


def base_normalize(abs_data, base_data, method="divide"):
    """Normalize values by a baseline
    Parameters
    ----------
    base_data : int, float
        Baseline data.
    abs_data : int, float
        Data to be normalized.
    method : str
        Method for normalization. Can be "divide" or "subtract".
    Returns
    ----------
    norm_data: int, float
        Normalized data.
    """
    base_cent = np.median(base_data)
    if method == "subtract":
        norm_data = abs_data - base_cent
    else:
        # to avoid NaNs
        if base_cent == 0:
            norm_data = 0
        else:
            norm_data = abs_data / base_cent
    return norm_data


def error_aug(
    df,
    id_col="Signal",
    gt_id="ECG",
    group_col="Participant",
    possible_info_columns=[
        "Signal",
        "Participant",
        "Task",
        "Rest",
        "SubSegIdx",
        "Method",
    ],
):
    # ECG-IEMe: Feature value obtained from ECG
    # for each observation minus the IEM error from all observations per participant
    info_columns = [col for col in df.columns if col in possible_info_columns]
    index_cols = [col for col in info_columns if col != id_col]
    new_sig_dfs = []
    all_ids = df[id_col].unique()
    # check that there is ground truth data (e.g. ECG) as well as at least
    # one other type of data (e.g. IEM) to get the error
    if len(all_ids) == 1 or gt_id not in all_ids:
        return df
    for other in [other_id for other_id in all_ids if other_id != gt_id]:
        new_sig_name = gt_id + "-" + other + "e"
        other_df = df[df[id_col] == other].drop(columns=[id_col])
        gt_df = df[df[id_col] == gt_id].drop(columns=[id_col])
        for method in other_df["Method"].unique():
            sel_other_df = other_df[other_df["Method"] == method].copy()
            sel_gt_df = gt_df.copy()
            sel_gt_df["Method"] = method
            sel_gt_df = sel_gt_df.set_index(index_cols)
            sel_other_df = sel_other_df.set_index(index_cols)
            e_df = sel_gt_df.sub(sel_other_df, axis="index")
            gt_recs = sel_gt_df.groupby([group_col]).apply(dict)
            e_recs = e_df.groupby([group_col]).apply(dict)

            for ind in gt_recs.index:
                sub_dfs = []
                gt_rec = gt_recs[ind]
                for k in gt_rec.keys():
                    ds = []
                    e_vals = e_recs[ind][k].to_list()
                    for r in gt_rec[k].reset_index().to_dict("records"):
                        for i in range(len(e_vals)):
                            d = r.copy()
                            d[k] = r[k] - e_vals[i]
                            d["SubSegIdx"] = i + 1
                            d["Signal"] = new_sig_name
                            ds.append(d)
                    if len(ds) > 0:
                        sub_df = pd.DataFrame(ds)
                        sub_dfs.append(sub_df)
                new_sig_df = pd.concat(sub_dfs, axis=1)
                new_sig_df = new_sig_df.loc[
                    :, ~new_sig_df.apply(lambda x: x.duplicated(), axis=1).all()
                ]
                new_sig_dfs.append(new_sig_df)
    new_df = pd.concat(new_sig_dfs)
    comb_df = pd.concat([df, new_df]).sort_values(by=info_columns)
    return comb_df
