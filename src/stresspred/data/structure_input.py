# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:24:55 2022

@author: danie
"""

import numpy as np
import pandas as pd

from stresspred.preprocess.transform import base_normalize, error_aug
from stresspred.preprocess.stress_feat_extract import stress_feat_extract


def signal_to_feat_df(
    in_df, feat_func=stress_feat_extract, include_error_aug=True, expected_columns=[]
):
    possible_info_columns = [
        "Signal",
        "Participant",
        "Task",
        "Rest",
        "SubSegIdx",
        "Method",
    ]
    info_columns = [
        col for col in in_df.columns if col in possible_info_columns]
    array_columns = ["Ibi", "IbiTime"]
    out_df = in_df.groupby(info_columns).apply(
        lambda x: feat_func(
            np.array(x[array_columns[0]]),
            np.array(x[array_columns[1]]),
            data_format="rri",
            check_successive=True,
            expected_columns=expected_columns,
        )
    )
    out_columns = list(out_df.columns)
    all_columns = info_columns.copy()
    all_columns.extend(out_columns)
    out_df = out_df.reset_index().loc[:, all_columns]
    if include_error_aug:
        return aug_df_with_error(out_df)
    return out_df


def ensure_columns_df(
    in_df,
    columns=["Signal", "Participant", "Task", "Rest", "SubSegIdx", "Method"],
    missing="NotGiven",
):
    for col in columns:
        if col not in in_df.columns:
            in_df[col] = missing
    return in_df


def create_base_df(in_df):
    """Create dataframe with segments to be used as baseline data
    Parameters
    ----------
    in_df : pd.DataFrame
        Original dataframe containing the following labels:
        "Signal","Participant","Task","Rest","SubSegIdx"
    Returns
    ----------
    out_df: pd.DataFrame
        New dataframe containing the rest segments for all other tasks
        now labeled as "Rest" = 2 to indicate that they should be used
        as basline data for that task.
    """
    list_dfs = []
    # different signal types, participants, and tasks should be
    # baseline normalized separately
    copy_in_df = in_df.copy()
    copy_in_df = ensure_columns_df(copy_in_df)

    if "Rest" not in copy_in_df["Rest"].unique():
        rest_val = copy_in_df["Rest"].unique()[0]
    else:
        rest_val = "Rest"
    for signal_index in copy_in_df["Signal"].unique():
        for participant in copy_in_df["Participant"].unique():
            for task in copy_in_df["Task"].unique():
                for method in copy_in_df["Method"].unique():
                    # take the rest segments that proceeded
                    # all other tasks except for the current task
                    baseline_df_all_subsegs = copy_in_df.loc[
                        (copy_in_df["Signal"] == signal_index)
                        & (copy_in_df["Participant"] == participant)
                        & (copy_in_df["Method"] == method)
                        & (copy_in_df["Task"] != task)
                        & (copy_in_df["Rest"] == rest_val)
                    ].copy()
                    count = 1
                    # in case there are multiple segments associated with a task
                    # (which was done when experimenting with using shorter segments
                    # for real-time classification), treat each one as a separate segment
                    # so that the features can be extracted from these segments separately
                    # and then later summarized (e.g. averaged) for normalization
                    # by one baseline feature value
                    for base_task in baseline_df_all_subsegs["Task"].unique():
                        for subseg_index in baseline_df_all_subsegs[
                            "SubSegIdx"
                        ].unique():
                            baseline_df = baseline_df_all_subsegs.loc[
                                (baseline_df_all_subsegs["Task"] == base_task)
                                & (baseline_df_all_subsegs["SubSegIdx"] == subseg_index)
                            ].copy()
                            baseline_df.loc[:, "SubSegIdx"] = count
                            baseline_df.loc[:, "Task"] = task
                            # to indicate this is to be used as baseline data
                            # rather than to classify as stress or rest
                            baseline_df.loc[:, "Rest"] = 2
                            list_dfs.append(baseline_df)
                            count += 1
    if len(list_dfs) > 1:
        out_df = pd.concat(list_dfs, ignore_index=True)
    elif len(list_dfs) == 1:
        out_df = list_dfs[0]
    else:
        out_df = baseline_df_all_subsegs
    return out_df


def norm_df(abs_feat_df, base_feat_df):
    possible_group_columns = ["Signal", "Participant", "Task", "Method"]
    group_columns = [
        col for col in abs_feat_df.columns if col in possible_group_columns
    ]
    possible_info_columns = [
        "Signal",
        "Participant",
        "Task",
        "Rest",
        "SubSegIdx",
        "Method",
    ]
    info_columns = [
        col for col in abs_feat_df.columns if col in possible_info_columns]
    intersect_columns = list(
        set(abs_feat_df.columns).intersection(set(base_feat_df.columns))
    )
    feat_columns = [
        col for col in intersect_columns if col not in info_columns]
    gb_abs = abs_feat_df.groupby(group_columns)
    keys_abs = [key for key, _ in gb_abs]
    parameters = []
    for key in keys_abs:
        parameters.append([(x, y) for x, y in zip(group_columns, key)])
    list_df = []
    for parameter in parameters:
        query = " and ".join(
            [
                "{0}=='{1}'".format(x[0], x[1])
                if type(x[1]) == str
                else "{0}=={1}".format(x[0], x[1])
                for x in parameter
            ]
        )
        a = abs_feat_df.query(query).reset_index(drop=True)
        b = base_feat_df.query(query).reset_index(drop=True)
        for i in range(len(a)):
            a_row = a.copy().iloc[[i]]
            for out_c in feat_columns:
                a_row[out_c] = base_normalize(a_row[out_c], b[out_c])
            list_df.append(a_row)
    norm_feat_df = pd.concat(list_df).reset_index(drop=True)
    return norm_feat_df


def aug_df_with_error(
    in_df,
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
    return error_aug(
        ensure_columns_df(in_df, columns=possible_info_columns),
        id_col=id_col,
        gt_id=gt_id,
        group_col=group_col,
        possible_info_columns=possible_info_columns,
    )


def prep_df_for_class(in_df, dropna=True, inf_is_missing=True, selected_features=None):

    info_columns = ["Signal", "Participant",
                    "Task", "Rest", "SubSegIdx", "Method"]
    y_column = "Rest"
    sub_column = "Participant"
    task_column = "Task"
    signal_column = "Signal"
    method_column = "Method"
    subseg_column = "SubSegIdx"

    class_df = in_df.copy()
    class_df = ensure_columns_df(class_df)

    if inf_is_missing:
        class_df = class_df.replace([np.inf, -np.inf], np.nan)
    if dropna:
        class_df = class_df.dropna(axis=1)
    X_data = class_df[class_df.columns[~class_df.columns.isin(
        info_columns)]].copy()
    if selected_features is not None:
        missing_features = [
            col for col in selected_features if col not in X_data.columns
        ]
        for feat in missing_features:
            X_data[feat] = np.nan
        X_data = X_data.loc[:, selected_features]

    y_data = class_df.loc[:, class_df.columns == y_column].values.ravel()
    sub_data = class_df.loc[:, class_df.columns == sub_column].values.ravel()
    task_data = class_df.loc[:, class_df.columns == task_column].values.ravel()
    signal_data = class_df.loc[:, class_df.columns ==
                               signal_column].values.ravel()
    method_data = class_df.loc[:, class_df.columns ==
                               method_column].values.ravel()
    subseg_data = class_df.loc[:, class_df.columns ==
                               subseg_column].values.ravel()
    return X_data, y_data, sub_data, task_data, signal_data, method_data, subseg_data
