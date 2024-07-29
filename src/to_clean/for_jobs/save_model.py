import os
import pathlib
import sys
import argparse
import onnxmltools
from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

from skl2onnx.common.data_types import FloatTensorType
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib

parser = argparse.ArgumentParser(
    description="Save default prediction pipeline trained on AUDACE dataset"
)
parser.add_argument(
    "--out_path",
    default=None,
    help="base path to .pkl and .onnx files where model will be saved",
)
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
    from stresspred import AudaceDataLoader, make_prediction_pipeline

    out_path = args.out_path
    root_in_path = args.root_in_path

    if out_path is None:
        out_path = "default_pred_pipe"

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

    # Use original feature values
    # rather than those relative to each participant's baseline
    rel_values = False
    # Use spreadsheet with already extracted features
    load_from_file = True
    # Do not save spreadsheet with features
    save_file = False

    # set location of root directory of input data
    train_loader = AudaceDataLoader(root=root_in_path)

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
    )

    # scale the data across participants based on the training set mean and SD
    scl = StandardScaler()

    # set the random seed
    seed = 0
    # classifier is XGBoost
    est = XGBClassifier(random_state=seed)

    # create sklearn prediction pipeline
    pred_pipe = make_prediction_pipeline(scl=scl, est=est)

    # Take all Neurokit2 time-domain, frequency-domain, and nonlinear
    # heart rate variability features
    # Neurokit2 features are all prefixed by "HRV_"
    nk_feats = [col for col in out_train["X"].columns if "HRV_" in col]
    selected_features = nk_feats

    # Train model
    pred_pipe.fit(out_train["X"].loc[:, selected_features], out_train["y"])

    # Save settings model was trained with
    pred_pipe.input_settings = {
        "selection_dict": selection_dict,
        "rel_values": rel_values,
    }
    # Save model to pkl
    pkl_path = out_path + ".pkl"
    joblib.dump(pred_pipe, pkl_path, compress=9)

    xgboost_model = pred_pipe.named_steps["estimator"]

    scaler = pred_pipe.named_steps["scaler"]

    # Specify path for onnx
    onnx_path = out_path + ".onnx"

    xgboost_model = pred_pipe.named_steps["estimator"]
    scaler = pred_pipe.named_steps["scaler"]

    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    # Convert the model to ONNX format
    onnx_model = convert_sklearn(
        pred_pipe,
        "ONNX model",
        [("input", FloatTensorType([None, len(selected_features)]))],
    )

    # Save the ONNX model to a file
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
