import os
import pathlib
import sys
import argparse
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
# imports for f1 score, sensitivity, specificity
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

import pandas as pd

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
    "--baseline_label", default=None, help="baseline label e.g. No"
)
parser.add_argument(
    "--model_label", default=None, help="model label e.g. XGBoost"
)
parser.add_argument(
    "--selected_feature_label", default=None, help="feature label e.g. All Neurokit2 features"
)
parser.add_argument(
    "--sig_name_train", default=None, help="train signal e.g. ECG-IEMLe + ECG"
)
parser.add_argument(
    "--sig_name_val", default=None, help="val signal e.g. IEML"
)

args = parser.parse_args()



if __name__ == "__main__":

    # load code in "stresspred" directory like a package
    code_paths = {}
    code_paths["repo_name"] = "p5-stress-classifier"

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
        make_search_space,
        sensitivity_score,
        specificity_score,
    )

    root_out_path = args.root_out_path
    root_in_path = args.root_in_path
    baseline_label = args.baseline_label
    model_label = args.model_label
    selected_feature_label = args.selected_feature_label
    sig_name_train = args.sig_name_train
    sig_name_val = args.sig_name_val

    if root_out_path is None:
        root_out_path = pathlib.Path(
            code_paths["repo_path"], "local_data", "outputs", "audace_analysis"
        )
        root_out_path.mkdir(parents=True, exist_ok=True)

    results_df_name = "results_paramopt_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

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
        load_from_file = True
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
        
        if baseline_label is None:
            baseline_labels = ["Yes", "No"]
        else:
            baseline_labels = [baseline_label]
        
        if model_label is None:
            model_labels = ["Log. Reg.", "XGBoost"]
        else:
            model_labels = [model_label]
            
        if selected_feature_label is None:
            selected_feature_labels = [
                "MedianNN",
                "MedianNN + RMSSD",
                "All Neurokit2 features",
            ]
        else:
            selected_feature_labels = [selected_feature_label]

        evaluation_labels = [
            "Cross-val on AUDACE",
        ]

        if sig_name_train is None:
            list_sig_name_train = ["ECG-IEMLe + ECG", "ECG", "IEML"]
        else:
            list_sig_name_train = [sig_name_train]
            
        if sig_name_val is None:
            list_sig_name_val = ["ECG", "IEML", "ECG-IEMLe"]
        else:
            list_sig_name_val = [sig_name_val]
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
                                    search_space = make_search_space(est=est)
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
                                        outer_cv, inner_cv = get_cv_iterator(
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
                                        scores = []
                                        f1_scores = []
                                        recall_scores = []
                                        precision_scores = []
                                        sensitivity_scores = []
                                        specificity_scores = []
                                        cv_count = 0
                                        for train, test in outer_cv:
                                            cv = inner_cv[cv_count]
                                            cv_count += 1
                                            search = GridSearchCV(pred_pipe, param_grid=search_space, cv=cv, verbose=3)
                                            search.fit(X=X, y=y)
                                            pred_pipe = pred_pipe.set_params(**search.best_params_)
                                            
                                            X_train = X.iloc[train]
                                            y_train = y[train]

                                            pred_pipe.fit(X_train, y_train)
                                            X_test = X.iloc[test]
                                            y_test = y[test]
                                            scores.append(
                                                pred_pipe.score(X_test, y_test)
                                            )
                                            f1_scores.append(
                                                f1_score(
                                                    y_test,
                                                    pred_pipe.predict(X_test),
                                                    pos_label="Stress",
                                                )
                                            )
                                            recall_scores.append(
                                                recall_score(
                                                    y_test,
                                                    pred_pipe.predict(X_test),
                                                    pos_label="Stress",
                                                )
                                            )
                                            precision_scores.append(
                                                precision_score(
                                                    y_test,
                                                    pred_pipe.predict(X_test),
                                                    pos_label="Stress",
                                                )
                                            )
                                            sensitivity_scores.append(
                                                sensitivity_score(
                                                    y_test,
                                                    pred_pipe.predict(X_test),
                                                    pos_label="Stress"
                                                )
                                            )
                                            specificity_scores.append(
                                                specificity_score(
                                                    y_test,
                                                    pred_pipe.predict(X_test),
                                                    pos_label="Stress",
                                                )
                                            )

                                        mean_accuracy = np.mean(scores)
                                        std_accuracy = np.std(scores)
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