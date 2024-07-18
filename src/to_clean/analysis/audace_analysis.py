import os
import pathlib
import sys
import argparse
import re
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(
    description="Save default prediction pipeline trained on AUDACE dataset"
)
parser.add_argument(
    "--root_out_path", default=None, help="location where output files should be saved"
)
parser.add_argument(
    "--root_in_path", default=None, help="location of root directory of input data"
)

parser.add_argument(
    "--include_sensitivity", default=True, help="whether to include sensitivity analysis"
)

parser.add_argument(
    "--include_specificity", default=True, help="whether to include specificity analysis"
)

args = parser.parse_args()

if __name__ == "__main__":

    # load code in "stresspred" directory like a package
    code_paths = {}
    code_paths["repo_name"] = "in-ear-stress"

    code_paths["repo_path"] = os.getcwd()
    base_dir = os.path.basename(code_paths["repo_path"])
    while base_dir != code_paths["repo_name"]:
        code_paths["repo_path"] = os.path.dirname(
            os.path.abspath(code_paths["repo_path"])
        )
        base_dir = os.path.basename(code_paths["repo_path"])
    package_dir = pathlib.Path(code_paths["repo_path"], "src")
    sys.path.append(str(package_dir))
    from stresspred import (
        AudaceDataLoader,
        P5_StressDataLoader,
        make_prediction_pipeline,
        get_cv_iterator,
    )

    root_out_path = args.root_out_path
    root_in_path = args.root_in_path
    include_sensitivity = args.include_sensitivity
    include_specificity = args.include_specificity

    if root_out_path is None:
        root_out_path = pathlib.Path(
            code_paths["repo_path"], "local_data", "outputs", "audace_analysis"
        )
        root_out_path.mkdir(parents=True, exist_ok=True)

    results_df_name = "all_dataset_results_20230414_nk022.csv"
    results_df_path = pathlib.Path(root_out_path, results_df_name)
    if not pathlib.Path(results_df_path).is_file():
        # Use ECG + ECG augmented with error between ECG and IEML features
        # For more information on how ECG-IEMLe was generated see error_aug()
        # in .structure_input
        selected_signals = ["ECG", "ECG-IEMLe"]
        # Use data collected before and during cold pressor test and mental task
        selected_tasks = ["CPT", "MENTAL"]
        selection_dict = {
            "Task": selected_tasks,
            "Signal": selected_signals,
        }
        
        # set location of root directory of input data
        train_loader = AudaceDataLoader(root=root_in_path)
        
        in_data = train_loader.get_ibi_df()
        
        # Use original feature values
        # rather than those relative to each participant's baseline
        rel_values = False
        # Do not use spreadsheet with already extracted features
        load_from_file = False
        # Save spreadsheet with features
        save_file = True

        # Load "out_train" dictionary containing:
        # X: all possible input features
        # y: ground truth stress/rest labels
        # method: method used for heartbeat extraction
        # signal: type of heartbeat signal (ECG, IEML, etc.)
        # sub: participant labels
        # task: task labels (rest periods are labeled with the task that came after)
        # subseg_data: if there are multiple segments corresponding to the same
        # set of labels (e.g. with smaller segment lengths or error augmentation),
        # their indices
        out_train = train_loader.get_split_pred_df(
            selection_dict=selection_dict,
            load_from_file=load_from_file,
            save_file=save_file,
            rel_values=rel_values,
            in_data=in_data,
        )

        out_train["dataset"] = np.array(
            [train_loader.dataset_label] * len(out_train["sub"])
        )

        rel_values = True
        out = AudaceDataLoader(root=root_in_path).get_split_pred_df(
            selected_tasks=["MENTAL", "CPT"],
            selected_signals=["ECG", "IEML"],
            load_from_file=True,
            save_file=False,
            rel_values=rel_values,
        )
        seed = 0
        nk_feats = [col for col in out["X"].columns if "HRV_" in col]

        baseline_labels = ["Yes", "No"]
        model_labels = ["Log. Reg.", "XGBoost"]
        selected_feature_labels = [
            "MedianNN",
            "MedianNN + RMSSD",
            "All Neurokit2 features",
        ]

        evaluation_labels = [
            "Cross-val on AUDACE",
        ]

        list_sig_name_train = ["ECG-IEMLe + ECG", "ECG", "IEML"]
        list_sig_name_val = ["ECG", "IEML", "ECG-IEMLe"]
        selected_tasks = ["MENTAL", "CPT"]
        list_iem_hb_methods = ["Auto-CRITIAS_BP"]
        first = True
        mode = "w"
        header = True
        eval_ind = 0
        for eval_ind in [0]:
            for iem_hb_method in list_iem_hb_methods:
                selected_methods = ["NotGiven", iem_hb_method]
                method_name_dict = {
                    "ECG": "Manual",
                    "IEML": iem_hb_method,
                    "ECG-IEMLe": iem_hb_method,
                }
                for est in model_labels:
                    for selected_features in selected_feature_labels:
                        for sig_name_train in list_sig_name_train:
                            sig_names_train = sig_name_train.split(" + ")
                            for sig_name_val in list_sig_name_val:
                                all_results = []
                                for baseline_label in baseline_labels:
                                    rel_values = baseline_label == "Yes"
                                    pred_pipe = make_prediction_pipeline(est=est)
                                    all_sig_names = np.concatenate(
                                        (
                                            np.array(sig_names_train),
                                            np.array([sig_name_val]),
                                        )
                                    )

                                    if eval_ind in [0, 1]:
                                        selected_signals = list(
                                            np.unique(all_sig_names)
                                        )
                                        if eval_ind == 0:
                                            loader = AudaceDataLoader(root=root_in_path)
                                            selection_dict = {
                                                "Task": selected_tasks,
                                                "Signal": selected_signals,
                                            }
                                        elif eval_ind == 1:
                                            loader = P5_StressDataLoader()
                                            selection_dict = {
                                                "Task": selected_tasks,
                                                "Signal": selected_signals,
                                                "Method": selected_methods,
                                            }
                                        out = loader.get_split_pred_df(
                                            selection_dict=selection_dict,
                                            selected_features=selected_features,
                                            load_from_file=True,
                                            save_file=False,
                                            rel_values=rel_values,
                                        )
                                        X = out["X"]
                                        y = out["y"]
                                        sub = out["sub"]
                                        task = out["task"]
                                        signal = out["signal"]
                                        outer_cv, _ = get_cv_iterator(
                                            sub,
                                            n_outer_splits=5,
                                            n_inner_splits=4,
                                            train_bool=np.array(
                                                [
                                                    True
                                                    if np.any(
                                                        sig == np.array(sig_names_train)
                                                    )
                                                    else False
                                                    for sig in signal
                                                ]
                                            ),
                                            val_bool=signal == sig_name_val,
                                        )
                                        accuracy_scores = []
                                        f1_scores = []
                                        recall_scores = []
                                        precision_scores = []
                                        sensitivity_scores = []
                                        specificity_scores = []
                                        for train, test in outer_cv:
                                            X_train = X.iloc[train]
                                            y_train = y[train]

                                            pred_pipe.fit(X_train, y_train)
                                            X_test = X.iloc[test]
                                            y_test = y[test]
                                            accuracy_scores.append(
                                                pred_pipe.score(X_test, y_test)
                                            )
                                            y_pred = pred_pipe.predict(X_test)
                                            f1_scores.append(
                                                f1_score(y_test, y_pred, average="macro")
                                            )
                                            recall_scores.append(
                                                recall_score(
                                                    y_test, y_pred, average="macro"
                                                )
                                            )
                                            precision_scores.append(
                                                precision_score(
                                                    y_test, y_pred, average="macro"
                                                )
                                            )
                                            tn, fp, fn, tp = confusion_matrix(
                                                y_test, y_pred
                                            ).ravel()
                                            sensitivity_scores.append(tp / (tp + fn))
                                            specificity_scores.append(tn / (tn + fp))
                                        mean_accuracy = np.mean(accuracy_scores)
                                        std_accuracy = np.std(accuracy_scores)
                                        mean_f1 = np.mean(f1_scores)
                                        std_f1 = np.std(f1_scores)
                                        mean_recall = np.mean(recall_scores)
                                        std_recall = np.std(recall_scores)
                                        mean_precision = np.mean(precision_scores)
                                        std_precision = np.std(precision_scores)
                                        mean_sensitivity = np.mean(sensitivity_scores)
                                        std_sensitivity = np.std(sensitivity_scores)
                                        mean_specificity = np.mean(specificity_scores)
                                        std_specificity = np.std(specificity_scores)
                                    else:
                                        selected_signals = sig_names_train
                                        selection_dict = {
                                            "Task": selected_tasks,
                                            "Signal": selected_signals,
                                        }
                                        out = AudaceDataLoader(root=root_in_path).get_split_pred_df(
                                            selection_dict=selection_dict,
                                            load_from_file=True,
                                            save_file=False,
                                            rel_values=rel_values,
                                        )
                                        X_train = out["X"]
                                        y_train = out["y"]
                                        sub_train = out["sub"]
                                        task_train = out["task"]
                                        signal_train = out["signal"]

                                        selected_signals = [sig_name_val]
                                        selection_dict = {
                                            "Task": selected_tasks,
                                            "Signal": selected_signals,
                                            "Method": selected_methods,
                                        }
                                        out = P5_StressDataLoader().get_split_pred_df(
                                            selection_dict=selection_dict,
                                            load_from_file=True,
                                            save_file=False,
                                            rel_values=rel_values,
                                        )

                                        X_test = out["X"]
                                        y_test = out["y"]
                                        sub_test = out["sub"]
                                        task_test = out["task"]
                                        print(
                                            X_train.columns[
                                                X_train.isna().any()
                                            ].tolist()
                                        )
                                        pred_pipe.fit(X_train, y_train)
                                        y_pred = pred_pipe.predict(X_test)
                                        mean_accuracy = accuracy_score(y_test, y_pred)
                                        std_accuracy = 0
                                    accuracy_perc_str = (
                                        str(np.round(mean_accuracy * 100, 1))
                                        + " ± "
                                        + str(np.round(std_accuracy * 100, 1))
                                        + " %"
                                    )
                                    f1_perc_str = (
                                        str(np.round(mean_f1 * 100, 1))
                                        + " ± "
                                        + str(np.round(std_f1 * 100, 1))
                                        + " %"
                                    )
                                    recall_perc_str = (
                                        str(np.round(mean_recall * 100, 1))
                                        + " ± "
                                        + str(np.round(std_recall * 100, 1))
                                        + " %"
                                    )
                                    precision_perc_str = (
                                        str(np.round(mean_precision * 100, 1))
                                        + " ± "
                                        + str(np.round(std_precision * 100, 1))
                                        + " %"
                                    )
                                    sensitivity_perc_str = (
                                        str(np.round(mean_sensitivity * 100, 1))
                                        + " ± "
                                        + str(np.round(std_sensitivity * 100, 1))
                                        + " %"
                                    )
                                    specificity_perc_str = (
                                        str(np.round(mean_specificity * 100, 1))
                                        + " ± "
                                        + str(np.round(std_specificity * 100, 1))
                                        + " %"
                                    )

                                    res_dict = {
                                        "Evaluation strategy": evaluation_labels[
                                            eval_ind
                                        ],
                                        "Signal train": sig_name_train
                                        + ": "
                                        + method_name_dict[sig_names_train[0]],
                                        "Signal val": sig_name_val
                                        + ": "
                                        + method_name_dict[sig_name_val],
                                        "Model": est,
                                        "Features": selected_features,
                                        "Baselined": baseline_label,
                                        "Accuracy": accuracy_perc_str,
                                        "F1": f1_perc_str,
                                        "Recall": recall_perc_str,
                                        "Precision": precision_perc_str,
                                        "Sensitivity": sensitivity_perc_str,
                                        "Specificity": specificity_perc_str,
                                    }
                                    all_results.append(res_dict)
                                result_df = pd.DataFrame(
                                    all_results
                                )  # .drop_duplicates().reset_index(drop=True)

                                result_df.to_csv(
                                    results_df_path,
                                    index=None,
                                    mode=mode,
                                    header=header,
                                )
                                if first:
                                    first = False
                                    header = None
                                    mode = "a"
    result_df = pd.read_csv(results_df_path)
    result_df_for_table = result_df.copy()

    unique_values = np.unique(
        np.concatenate(
            [
                np.array(result_df_for_table[col].unique())
                for col in result_df_for_table.columns
            ]
        )
    )
    replace_dict = {}
    for val in unique_values:
        for suf in [": Manual", ": Auto", " %"]:
            if suf in str(val):
                replace_dict[str(val)] = str(val).split(suf)[0]
    orig_name_syn_data = "ECG-IEMLe"
    name_syn_data = "SYN"
    
    # orig_name_aug_data = orig_name_syn_data + " + ECG"
    name_aug_data = "ECG & " + name_syn_data
    replace_dict["ECG-IEMLe + ECG: Auto-CRITIAS_BP"] = name_aug_data

    for k, v in replace_dict.items():
        if orig_name_syn_data in v:
            replace_dict[k] = re.sub(orig_name_syn_data, name_syn_data, v)
        
    orig_name_ieml = "IEML"
    name_ieml = "IEM"
    for k, v in replace_dict.items():
        if orig_name_ieml in v:
            replace_dict[k] = re.sub(orig_name_ieml, name_ieml, v)
    
    orig_name_feat_set_b = "MedianNN + RMSSD"
    name_feat_set_b = "MedianNN & RMSSD"
    
    replace_dict[orig_name_feat_set_b] = name_feat_set_b
            
    replace_dict["All Neurokit2 features"] = "All HRV features"
    
    rename_dict = {}
    val_col_name = "Test"
    acc_col_name = "Acc. (%)"
    est_col_name = "Classifier"
    base_col_name = "Baselining"
    sensitivity_col_name = "Sens. (%)"
    specificity_col_name = "Spec. (%)"
    rename_dict["Signal val"] = val_col_name
    rename_dict["Signal train"] = "Train"
    rename_dict["Accuracy"] = acc_col_name
    rename_dict["Model"] = est_col_name
    rename_dict["Baselined"] = base_col_name
    rename_dict["Sensitivity"] = sensitivity_col_name
    rename_dict["Specificity"] = specificity_col_name
    
    result_df_for_table = result_df_for_table.replace(replace_dict).rename(
        columns=rename_dict
    )
    result_df_for_table["Features"] = pd.Categorical(
        result_df_for_table["Features"],
        ["MedianNN", name_feat_set_b, "All HRV features"],
    )

    def reindent(s, num_spaces=4):
        indent = " " * num_spaces
        return s.replace("\n", "\n" + indent)

    def write_latex_table(df, out_path=None):
        s = (
            r"""
\begin{table*}[]
    \centering
"""
            + " " * 4
            + reindent(df.style.hide(axis="index").to_latex(hrules=True), 8)
            + r"""
    \caption{}
    \label{tab:my_label}
\end{table*}
        """
        )
        if out_path is None:
            return s
        else:
            # if parent path does not exist, create it
            pathlib.Path(out_path).resolve().parent.mkdir(parents=True, exist_ok=True)
            f = open(out_path, "w")
            f.write(s)
            f.close()
    
    result_table_cols = ["Train", val_col_name, base_col_name, "Features", est_col_name, acc_col_name]
    if include_sensitivity:
        result_table_cols.append(sensitivity_col_name)
    if include_specificity:
        result_table_cols.append(specificity_col_name)
    result_df_for_table = result_df_for_table.loc[:, result_table_cols].sort_values(
        by=result_table_cols
    )
    tables_for_latex = []
    latex_cols = ["Features", est_col_name, acc_col_name]
    if include_sensitivity:
        latex_cols.append(sensitivity_col_name)
    if include_specificity:
        latex_cols.append(specificity_col_name)
    #### TABLE 1: COMPARING BASELINED MODELS TRAINED ON ECG AND TESTED ON ECG
    table1 = result_df_for_table[
        (result_df_for_table["Train"] == "ECG") & 
        (result_df_for_table[val_col_name] == "ECG") &
        (result_df_for_table[base_col_name].isin(["Yes"]))
    ]
    tables_for_latex.append(table1)
    #write_latex_table(table1, out_path="table1.tex")
    print(table1.to_latex(index=False))
    
    #### TABLE 2: COMPARING NON-BASELINED MODELS TRAINED ON ECG AND TESTED ON ECG
    table2 = result_df_for_table[
        (result_df_for_table["Train"] == "ECG") & 
        (result_df_for_table[val_col_name] == "ECG") &
        (result_df_for_table[base_col_name].isin(["No"]))
    ]
    tables_for_latex.append(table2)
    print(table2.to_latex(index=False))

    #### TABLE 3: COMPARING NON-BASELINED MODELS TRAINED ON IEML AND TESTED ON IEML
    table3 = result_df_for_table[
        (result_df_for_table["Train"].isin([name_ieml]))
        & (result_df_for_table[val_col_name].isin([name_ieml]))
        & (result_df_for_table[base_col_name].isin(["No"]))
    ]
    tables_for_latex.append(table3)
    print(table3.to_latex(index=False))
    
    
    #### TABLE 4: COMPARING NON-BASELINED MODELS TRAINED ON IEML AND TESTED ON ECG
    table4 = result_df_for_table[
        (result_df_for_table["Train"].isin([name_ieml]))
        & (result_df_for_table[val_col_name].isin(["ECG"]))
        & (result_df_for_table[base_col_name].isin(["No"]))
    ]
    tables_for_latex.append(table4)
    print(table4.to_latex(index=False))
    
    #### TABLE 5: COMPARING NON-BASELINED MODELS TRAINED ON IEML AND TESTED ON ECG_syn
    table5 = result_df_for_table[
        (result_df_for_table["Train"].isin([name_ieml]))
        & (result_df_for_table[val_col_name].isin([name_syn_data]))
        & (result_df_for_table[base_col_name].isin(["No"]))
    ]
    tables_for_latex.append(table5)
    print(table5.to_latex(index=False))
    
    #### TABLE 6: COMPARING NON-BASELINED MODELS TRAINED ON ECG AND TESTED ON IEML
    table6 = result_df_for_table[
        (result_df_for_table["Train"].isin(["ECG"]))
        & (result_df_for_table[val_col_name].isin([name_ieml]))
        & (result_df_for_table[base_col_name].isin(["No"]))
    ]
    tables_for_latex.append(table6)
    print(table6.to_latex(index=False))

    #### TABLE 6: COMPARING NON-BASELINED MODELS TRAINED ON AUGMENTED ECG
    table6 = result_df_for_table[
        (result_df_for_table["Train"].isin([name_aug_data]))
        & (result_df_for_table[val_col_name].isin(["ECG", name_ieml, name_syn_data]))
        & (result_df_for_table[base_col_name].isin(["No"]))
        & (result_df_for_table["Features"].isin(["All HRV features"]))
    ]
    print(table6.to_latex(index=False))
    
    for table_for_latex in tables_for_latex:
        table_for_latex = table_for_latex.loc[:, latex_cols].sort_values(
            by=latex_cols
        )
        print(table_for_latex.to_latex(index=False))

    abs_df, _ = AudaceDataLoader(root=root_in_path).get_feat_dfs()
    # abs_df = abs_df[abs_df["Task"].isin(["MENTAL", "CPT"])].copy().reset_index()
    color_other_sig_name = {"IEML": "#F79646", "ECG-IEMLe": "#7030A0"}
    task_names_for_plot = ["MENTAL", "CPT"]
    for hrv_col in ["HRV_MedianNN", "HRV_RMSSD"]:
        for other_sig_name in ["IEML", "ECG-IEMLe"]:
            gt_sig_name = "ECG"
            # other_sig_name = "IEML"

            if other_sig_name == "ECG-IEMLe":
                df = abs_df.copy()
                gt = "ECG"
                info_cols = ["Signal", "Participant", "Task", "Rest", "SubSegIdx", "Method"]
                grouping_col = "Signal"
                index_cols = [col for col in info_cols if col != grouping_col]
                new_sig_dfs = []
                for other in ["IEML"]:
                    new_sig_name = gt
                    gt_df = (
                        df[df[grouping_col] == gt]
                        .drop(columns=[grouping_col])
                        .set_index(index_cols)
                    )
                    other_df = (
                        df[df[grouping_col] == other]
                        .drop(columns=[grouping_col])
                        .set_index(index_cols)
                    )
                    e_df = gt_df.sub(other_df, axis="index")
                    gt_recs = gt_df.groupby(["Participant"]).apply(dict)
                    e_recs = e_df.groupby(["Participant"]).apply(dict)

                    for ind in gt_recs.index:
                        sub_dfs = []
                        gt_rec = gt_recs[ind]
                        for k in gt_rec.keys():
                            ds = []
                            e_vals = e_recs[ind][k].to_list()
                            for r in gt_rec[k].reset_index().to_dict("records"):
                                for i in range(len(e_vals)):
                                    d = r.copy()
                                    d[k] = r[k]
                                    d["SubSegIdx"] = i + 1
                                    d["Signal"] = new_sig_name
                                    ds.append(d)
                            sub_df = pd.DataFrame(ds)
                            sub_dfs.append(sub_df)
                        new_sig_df = pd.concat(sub_dfs, axis=1)
                        new_sig_dfs.append(new_sig_df)
                new_df = pd.concat(new_sig_dfs)
                gt_record = new_df.loc[
                    :, ~new_df.apply(lambda x: x.duplicated(), axis=1).all()
                ].copy()
            else:
                gt_record = abs_df[abs_df["Signal"] == gt_sig_name]
            record = abs_df[abs_df["Signal"] == other_sig_name]
            gt_record = gt_record[gt_record["Task"].isin(task_names_for_plot)]
            record = record[record["Task"].isin(task_names_for_plot)]

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
                        gt_sig_name: np.array(gt_record[hrv_col])[cond_ind_gt],
                        other_sig_name: np.array(record[hrv_col])[cond_ind],
                    }
                )
                dfs_list.append(df)
                conditions.append(cond)
            if other_sig_name == "IEML":
                other_sig_name_plot = name_ieml
            else:
                other_sig_name_plot = name_syn_data
            for i in range(len(dfs_list)):
                df = dfs_list[i]
                data1 = df[gt_sig_name]
                data2 = df[other_sig_name]
                sm.graphics.mean_diff_plot(
                    data1, data2, ax=axs[i], scatter_kwds={"alpha": 0.25, "color": color_other_sig_name[other_sig_name]}
                )

                axs[i].set(
                    xlabel="Mean of " + gt_sig_name + " and " + other_sig_name_plot,
                    ylabel=""
                    + gt_sig_name
                    + " - "
                    + other_sig_name_plot,
                )
                if hrv_col == "HRV_RMSSD":
                    axs[i].set(
                        ylim=[-1200, 400],
                        xlim=[0, 800]
                    )
                else:
                    axs[i].set(
                        ylim=[-300, 300],
                        xlim=[300, 1200]
                    )
                axs[i].set_title(label=conditions[i] + " conditions", fontsize=20)
            #title_txt = "Bland-Altman plots for " + hrv_col
            hrv_col_for_plot = hrv_col.split("HRV_")[1]
            title_txt = hrv_col_for_plot
            plt.suptitle(title_txt, y=1.05, fontsize=20)
            fig_name = "bland-altman_" + hrv_col_for_plot + "_" + other_sig_name_plot + ".png"
            fig_path = pathlib.Path(root_out_path, fig_name)
            fig.savefig(
                fig_path, facecolor="white", transparent=False, bbox_inches="tight"
            )
            plt.show()
    other = "IEML"
    id_col = "Signal"
    index_cols = [col for col in info_cols if col != id_col]
    gt_df = (
        abs_df[abs_df[id_col] == gt].copy().set_index(index_cols).drop(columns=[id_col])
    )
    other_df = (
        abs_df[abs_df[id_col] == other]
        .copy()
        .set_index(index_cols)
        .drop(columns=[id_col])
    )
    e_df = gt_df.sub(other_df, axis="index")

    plt.figure(figsize=(18, 5))
    plt.scatter(e_df["HRV_MedianNN"].values, e_df["HRV_RMSSD"].values)
    plt.show()

    coef_dfs = []
    for sig_name in ["ECG", "IEML"]:
        coefs = []
        selected_tasks = ["MENTAL", "CPT"]
        selected_signals = [sig_name]
        rel_values = False
        selection_dict = {
            "Task": selected_tasks,
            "Signal": selected_signals,
        }
        selected_features = ["HRV_MedianNN", "HRV_RMSSD"]
        out = AudaceDataLoader(root=root_in_path).get_split_pred_df(
            selection_dict=selection_dict,
            selected_features=selected_features,
            load_from_file=True,
            save_file=False,
            rel_values=rel_values,
        )
        X = out["X"]
        y = out["y"]
        sub = out["sub"]
        task = out["task"]
        signal = out["signal"]

        outer_cv, _ = get_cv_iterator(sub, n_outer_splits=5, n_inner_splits=4,)

        pred_pipe = make_prediction_pipeline()

        for train, test in outer_cv:
            X_train = X.iloc[train]
            y_train = y[train]
            pred_pipe.fit(X_train, y_train)
            coef_dict = dict(
                zip(
                    selected_features,
                    list(pred_pipe.named_steps["estimator"].coef_.flatten()),
                )
            )

            coefs.append(coef_dict)
        coef_df = pd.DataFrame(coefs)
        coef_df = pd.melt(coef_df, var_name="Feature", value_name="Coefficient")
        coef_df["Signal"] = sig_name
        coef_dfs.append(coef_df)
    coef_df = pd.concat(coef_dfs)
    
    coef_df = coef_df.replace({"IEML": name_ieml, "HRV_MedianNN": "MedianNN", "HRV_RMSSD": "RMSSD"})
    fig, axes  = plt.subplots(figsize=(3, 3), dpi=1000)
    
    box = sns.boxplot(
        x="Feature", y="Coefficient", hue="Signal", data=coef_df, palette="Set3", ax=axes
    )
    plt.ylim([-2.5, 2.5])
    fig_name = "Coefs_log_reg.png"
    fig_path = pathlib.Path(root_out_path, fig_name)
    fig.savefig(
        fig_path, facecolor="white", transparent=False, bbox_inches="tight"
    )
    ######## PLOT EXAMPLE TD IEM SEGMENTS
    loader = AudaceDataLoader(root=root_in_path)
    ibi_df = loader.get_ibi_df()
    
    fig, axs = plt.subplots(2, 1, figsize=(7, 3.5), dpi=1000)

for ax_ind in [0, 1]:

    data_format = "DB8k"
    if ax_ind==0:
        start_time = 180
    else:
        start_time = 246
    end_time = start_time + 15
    sig_name = "IEML"
    rri_time = ibi_df[(ibi_df["Signal"]==sig_name) & (ibi_df["Participant"] == loader.sub_label)].loc[:,["IbiTime"]].values
    rri = ibi_df[(ibi_df["Signal"]==sig_name) & (ibi_df["Participant"] == loader.sub_label)].loc[:,["Ibi"]].values

    peak_time = rri_time[(rri_time >= start_time) & (rri_time <= end_time)]
    sig_name = "ieml"
    sig_info = loader.get_sig(
        sig_name=sig_name, data_format=data_format, start_time=start_time, end_time=end_time
    )
    axs[ax_ind].plot(sig_info["time"], sig_info["sig"] - np.mean(sig_info["sig"]), label="In-ear audio", alpha=0.75)
    #plt.scatter(peak_time, sig_info["sig"][timestamp_to_samp(peak_time, sig_time=sig_info["time"])], color="orange", zorder=2)
    events = [peak_time]
    color = ["orange"]
    linestyle = ["--"]
    label = ["Heartbeat detected"]
    for i, event in enumerate(events):
        for j in events[i]:
            axs[ax_ind].axvline(j, color=color[i], linestyle=linestyle[i], label=label[i])

    # Display only one legend per event type
    handles, labels = axs[ax_ind].get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    if ax_ind==1:
        axs[ax_ind].legend(newHandles, newLabels, loc='upper center', bbox_to_anchor=(0.5, -0.8),
                  fancybox=True, shadow=True, ncol=5)
        axs[ax_ind].set_xlabel("Time (seconds)")
        #axs[ax_ind].set_ylim([-0.010, 0.003])
        axs[ax_ind].set_title("Stress condition")
    else:
        axs[ax_ind].set_title("Rest condition")
    axs[ax_ind].set_ylim([-0.005, 0.005])
    axs[ax_ind].set_xlim([start_time, end_time])
    axs[ax_ind].set_yticks([])
    #axs[ax_ind].get_yaxis().set_visible(False)
#plt.suptitle("Examples of in-ear audio recorded during rest and stress conditions")    
plt.tight_layout()
fig_name = "Example_TD_IEM.png"
fig_path = pathlib.Path(root_out_path, fig_name)
fig.savefig(
    fig_path, facecolor="white", transparent=False, bbox_inches="tight"
)
plt.show()
    
