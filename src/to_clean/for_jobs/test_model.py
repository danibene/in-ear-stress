import os
import pathlib
import sys
import numpy as np
import argparse
import pandas as pd
import joblib

parser = argparse.ArgumentParser(
    description="Stress classification with shorter segments"
)
parser.add_argument(
    "--saved_model_path", default=None, help="where trained model is saved"
)
parser.add_argument("--out_path", default=None, help="where to save outputs")
parser.add_argument(
    "--root_in_path", default=None, help="location of root directory of input data"
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
        code_paths,
        P5M5DataLoader,
        hb_extract,
        peak_time_to_rri,
        write_dict_to_json,
    )

    out_path = args.out_path

    saved_model_path = args.saved_model_path
    if saved_model_path is None:
        saved_model_path = "default_pred_pipe.pkl"
    root_in_path = args.root_in_path

    # set minimum and maximum beats per minute
    # for converting peaks detected to interbeat intervals
    min_bpm = 40
    max_bpm = 200

    # list to store interbeat intervals
    list_ibi_dfs = []

    # set location of root directory of input data
    test_loader = P5M5DataLoader(root=root_in_path)
    dataset_iterator = test_loader.get_dataset_iterator()
    # taking only last two samples so that this example can be run quickly
    # (remove this line to iterate over all datasets/participants/conditions)
    dataset_iterator = dataset_iterator[100:102]
    # methods for heartbeat extraction and signals
    rri_extraction = [
        {"method": "ecg_audio", "sig_name": "ecg_audio"},
        {"method": "nk_ppg_elgendi", "sig_name": "ti_ppg"},
        {"method": "temp", "sig_name": "ieml"},
    ]
    # iterate over all method and signal pairs
    for rri_ext in rri_extraction:
        # iterate over all datasets, participants, and conditions
        for dataset_label, sub_id, cond_label in dataset_iterator:
            method = rri_ext["method"]
            sig_name = rri_ext["sig_name"]
            #### STEP 1: LOAD & PREPROCESS SIGNAL
            # load data for requested dataset, participant, and condition
            test_loader.dataset_label = dataset_label
            test_loader.sub_id = sub_id
            test_loader.cond_label = cond_label
            # load raw signal resampled at 8 kHz
            sig_info = test_loader.get_sig(sig_name=sig_name, data_format="8k")

            #### STEP 2: HEARTBEAT EXTRACTION
            # extract hearbeat and get timestamps of peaks in seconds
            sig_info["peak_time"] = hb_extract(
                sig=sig_info["sig"],
                sig_time=sig_info["time"],
                sig_name=sig_info["name"],
                method=method,
            )

            # get interbeat intervals from peak times
            # removing any intervals smaller than minimum/larger than maximum
            sig_info["rri"], sig_info["rri_time"] = peak_time_to_rri(
                sig_info["peak_time"], min_rri=60000 / max_bpm, max_rri=60000 / min_bpm
            )
            sig_info["Task"] = cond_label
            sig_info["start_time"] = 0
            sig_info["end_time"] = np.inf

            segment_indices = np.where(
                (sig_info["rri_time"] > sig_info["start_time"])
                & (sig_info["rri_time"] <= sig_info["end_time"])
            )
            sig_info["Ibi"] = sig_info["rri"][segment_indices]
            sig_info["IbiTime"] = sig_info["rri_time"][segment_indices]
            sig_info["SubSegIdx"] = 1
            sig_info["Participant"] = dataset_label + "_" + str(sub_id)
            sig_info["Method"] = method
            sig_info["Rest"] = "NotGiven"
            # for compatibility with how the AUDACE data is formatted
            if "ECG" in sig_name.upper():
                sig_info["Signal"] = "ECG"
            else:
                sig_info["Signal"] = sig_name.upper().split("_")[-1]
            # columns of dataframe expected by feature extraction functions
            keep_keys = [
                "Signal",
                "Participant",
                "Task",
                "Rest",
                "SubSegIdx",
                "Ibi",
                "IbiTime",
                "Method",
            ]
            out_dict = {keep_key: sig_info[keep_key] for keep_key in keep_keys}
            out_df = pd.DataFrame(out_dict)
            list_ibi_dfs.append(out_df)
    ibi_df = pd.concat(list_ibi_dfs)
    # Use ECG + ECG augmented with error between ECG and IEML features
    # For more information on how ECG-IEMLe was generated see error_aug()
    # in .structure_input
    selected_signals = list(ibi_df["Signal"].unique())
    # Use data collected before and during cold pressor test and mental task
    selected_tasks = list(ibi_df["Task"].unique())
    selection_dict = {
        "Task": selected_tasks,
        "Signal": selected_signals,
    }

    #### STEP 3: HRV FEATURE EXTRACTION AND TRANSFORMATION
    # Use original feature values
    # rather than those relative to each participant's baseline
    rel_values = False
    # Do not use spreadsheet with already extracted features
    load_from_file = False
    # Do not save spreadsheet with features
    save_file = False

    # Load "out_test" dictionary containing:
    # X: all possible input features
    # y: ground truth stress/rest labels
    # method: method used for heartbeat extraction
    # signal: type of heartbeat signal (ECG, IEML, etc.)
    # sub: participant labels
    # task: task labels (rest periods are labeled with the task that came after)
    # subseg_data: if there are multiple segments corresponding to the same
    # set of labels (e.g. with smaller segment lengths or error augmentation),
    # their indices
    out_test = test_loader.get_split_pred_df(
        selection_dict=selection_dict,
        load_from_file=load_from_file,
        save_file=save_file,
        rel_values=rel_values,
        in_data=ibi_df,
    )

    #### STEP 4: CLASSIFICATION
    pred_pipe = joblib.load(saved_model_path)
    missing_features = [
        col for col in pred_pipe.feature_names_in_ if col not in out_test["X"].columns
    ]
    for feat in missing_features:
        out_test["X"][feat] = np.nan
    selected_features = pred_pipe.feature_names_in_
    # save y predictions
    out_test["y_pred"] = pred_pipe.predict(out_test["X"].loc[:, selected_features])

    # if output path is provided, save inputs with predictions as json file
    if out_path is not None:
        write_dict_to_json(out_test, out_path)
