#### STEP 1: LOAD SIGNAL
#### STEP 2: HEARTBEAT EXTRACTION
#### STEP 3: HRV FEATURE EXTRACTION AND TRANSFORMATION
#### STEP 4: CLASSIFICATION
import os
import pathlib
import sys
import numpy as np
import argparse
import pandas as pd
import joblib
import librosa

parser = argparse.ArgumentParser(
    description="Stress classification testing example with rest recording from `hum' dataset"
)
parser.add_argument(
    "--saved_model_path", default=None, help="where trained model is saved"
)
parser.add_argument("--out_path", default=None, help="where to save outputs")
parser.add_argument(
    "--root_in_path", default=None, help="location of root directory of input data"
)
parser.add_argument('--use_matlab', default=False, action=argparse.BooleanOptionalAction)



args = parser.parse_args()

if __name__ == "__main__":
    # load code in "stresspred" directory like a package
    code_paths = {}
    code_paths["repo_name"] = "p5-stress-classifier"

    code_paths["repo_path"] = pathlib.Path(__file__).parent.resolve()
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
        StressBioDataLoader,
        hb_extract,
        peak_time_to_rri,
        write_dict_to_json,
        sampling_rate_to_sig_time,
        resample_nonuniform
    )
    use_matlab = args.use_matlab
        
    out_path = args.out_path

    saved_model_path = args.saved_model_path
    if saved_model_path is None:
        saved_model_path = "default_pred_pipe.pkl"
    root_in_path = args.root_in_path

    # set minimum and maximum beats per minute
    # for converting peaks detected to interbeat intervals
    min_bpm = 40
    max_bpm = 200
    
    # set location of root directory of input data
    test_loader = StressBioDataLoader(root=root_in_path)
    # method for heartbeat extraction and associated signal
    # note that the "temp" method used here for is not the same as the method used
    # to extract the features used to train the model 
    # as that heartbeat extraction method is not publicly available
    method = "temp"
    sig_name = "ieml"
    
    if root_in_path is None:
        root_in_path = pathlib.Path(code_paths["repo_path"], "local_data", "test_model_hum_example")
    file_path = pathlib.Path(root_in_path,"sub-202102239_task-rest_IEML.wav")

    #### STEP 1: LOAD SIGNAL
    # load raw signal resampled at 8 kHz
    sig, sampling_rate = librosa.load(file_path, sr=None,
                          mono=True, offset=0)
    # preprocess signal
    new_sampling_rate = 8000
    sig_time = sampling_rate_to_sig_time(
        sig=sig, sampling_rate=sampling_rate)
    sig, sig_time = resample_nonuniform(
        sig,
        sig_time,
        new_sampling_rate=new_sampling_rate,
        use_matlab=use_matlab,
    )

    #### STEP 2: HEARTBEAT EXTRACTION
    # extract hearbeat and get timestamps of peaks in seconds
    sig_info = {}
    sig_info["peak_time"] = hb_extract(
        sig=sig,
        sig_time=sig_time,
        method=method,
    )

    # get interbeat intervals from peak times
    # removing any intervals smaller than minimum/larger than maximum
    sig_info["rri"], sig_info["rri_time"] = peak_time_to_rri(
        sig_info["peak_time"], min_rri=60000 / max_bpm, max_rri=60000 / min_bpm
    )
    sig_info["Task"] = "NotGiven"
    sig_info["start_time"] = 0
    sig_info["end_time"] = np.inf

    segment_indices = np.where(
        (sig_info["rri_time"] > sig_info["start_time"])
        & (sig_info["rri_time"] <= sig_info["end_time"])
    )
    sig_info["Ibi"] = sig_info["rri"][segment_indices]
    sig_info["IbiTime"] = sig_info["rri_time"][segment_indices]
    sig_info["SubSegIdx"] = 1
    sig_info["Participant"] = str(file_path)
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
    ibi_df = pd.DataFrame(out_dict)
    
    # resegment example data (which was 300 seconds) into 180-second segments 
    # because the model was trained on 180-second segments
    seg_len = 180
    hop_len = 90
    ibi_df = test_loader.resegment_ibi_df(
        seg_len=seg_len,
        hop_len=hop_len,
        in_data=ibi_df,
        max_minus_min_ibi_tol=2, # Should be within 2 seconds of 180-seconds
    )
    
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
    print("Predictions: ")
    print(out_test["y_pred"])

    # if output path is provided, save inputs with predictions as csv or json file
    if out_path is not None:
        if pathlib.Path(out_path).suffix == ".csv":
            # if parent path does not exist create it
            pathlib.Path(out_path).resolve().parent.mkdir(parents=True, exist_ok=True)
            out_test_df = pd.DataFrame({k:v for k, v in out_test.items() if k != "X"})
            out_test_df = pd.concat((out_test_df, out_test["X"]), axis=1)
            out_test_df.to_csv(out_path)
        else:
            write_dict_to_json(out_test, out_path)
