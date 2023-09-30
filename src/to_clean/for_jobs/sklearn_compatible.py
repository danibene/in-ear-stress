from sklearn.compose import make_column_selector
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from warnings import warn
from json_tricks import dump
import numpy as np
import pandas as pd
import scipy
import os
import soundfile as sf
import matplotlib.pyplot as plt
import pathlib
import joblib
from xgboost import XGBClassifier

import sys  # isort:skip

# fmt: off

code_paths = {}
code_paths["repo_name"] = "p5-stress-classifier"

code_paths["repo_path"] = os.getcwd()
base_dir = os.path.basename(code_paths["repo_path"])
while base_dir != code_paths["repo_name"]:
    code_paths["repo_path"] = os.path.dirname(
        os.path.abspath(code_paths["repo_path"]))
    base_dir = os.path.basename(code_paths["repo_path"])

package_dir = pathlib.Path(code_paths["repo_path"], "src")
sys.path.append(str(package_dir))
from stresspred import (code_paths,
                        peak_time_to_rri,
                        AudaceDataLoader,
                        make_prediction_pipeline,
                        write_dict_to_json,
                        stress_feat_extract,
                        hb_extract)
# in order to load the Neurokit submodule as a package
for path in code_paths["neurokit2_paths"]:  # isort:skip
    sys.path.insert(0, path)  # isort:skip

import neurokit2 as nk
from neurokit2.signal import signal_interpolate
from neurokit2.misc import find_successive_intervals
# fmt: on


duration = 180

records = []
for duration in [4, 90, 180]:
    ecg = nk.ecg_simulate(duration=duration)
    signals, info = nk.ecg_process(ecg)
    ecg = nk.ecg_simulate(duration=duration)
    _, info = nk.ecg_process(ecg)
    rri, rri_time = peak_time_to_rri(info["ECG_R_Peaks"] / info["sampling_rate"])
    d = {"rri": rri, "rri_time": rri_time}
    records.append(d)


def records_to_X_df(records, ind_str="_ind_", len_str="_len"):
    new_records = []
    for rec in records:
        new_rec = {}
        for k in rec.keys():
            # new_rec[k + len_str] = len(rec[k])
            for i in range(len(rec[k])):
                new_rec[k + ind_str + str(i)] = rec[k][i]
        new_records.append(new_rec)
    return pd.DataFrame(new_records)


def get_keys_from_cols(X_df, ind_str="_ind_"):
    d = {}
    keys = np.unique([col.split(ind_str)[0] for col in X_df.columns])
    for k in keys:
        d[k] = [col for col in X_df.columns if k + ind_str in col]
    return d


def X_df_to_records(X_df, ind_str="_ind_"):
    records = []
    key_d = get_keys_from_cols(X_df)
    for index, row in X_df.iterrows():
        rec = {}
        for k in key_d.keys():
            rec[k] = row.loc[key_d[k]].values[np.isfinite(row.loc[key_d[k]].values)]
        records.append(rec)
    return records


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        return self


class HeartbeatExtractor(CustomTransformer):
    """
    Class used to preprocess
    """

    def __init__(self):
        self.feature_names_out = None
        return None

    def transform(self, X=None):
        records = X_df_to_records(X)
        outs = []
        for rec in records:
            outs.append(hb_extract(**rec))
        all_out = pd.concat(outs).set_index(X.index)
        self.feature_names_out = list(all_out.columns)
        return all_out


class StressFeatExtractor(CustomTransformer):
    """
    Class used to preprocess
    """

    def __init__(self):
        self.feature_names_out = None
        return None

    def transform(self, X=None):
        records = X_df_to_records(X)
        outs = []
        for rec in records:
            outs.append(stress_feat_extract(**rec))
        all_out = pd.concat(outs).set_index(X.index)
        self.feature_names_out = list(all_out.columns)
        return all_out

    def get_feature_names_out(self):
        return self.feature_names_out


X_df = records_to_X_df(records)
y = ["Stress", "Rest", "Stress"]
feat_extraction = StressFeatExtractor()
seed = 0
pred_pipe = Pipeline(
    [
        ("feat_extraction", StressFeatExtractor()),
        ("est", XGBClassifier(random_state=seed)),
    ]
)
pred_pipe.fit(X_df, y)

pred_pipe.predict(X_df)
trained_model_path = "test.pkl"
joblib.dump(pred_pipe, trained_model_path, compress=9)
clf2 = joblib.load(trained_model_path)
clf2.predict(X_df)

import dill

dill.dump_session("test.pkl")


"""
class testEstimator(BaseEstimator,TransformerMixin):
    def __init__(self,string):
        self.string = string

    def fit(self,X):
        return self

    def transform(self,X):
        return np.full(X.shape, self.string).reshape(-1,1)

    def get_feature_names(self):
        return self.string

transformers = [('first_transformer',testEstimator('A'),1), ('second_transformer',testEstimator('B'),0)]
column_transformer = ColumnTransformer(transformers)
steps = [('scaler',RobustScaler()), ('transformer', column_transformer)]
pipeline = Pipeline(steps)

dt_test = np.zeros((1000,2))
pipeline.fit_transform(dt_test)

for name,step in pipeline.named_steps.items():
    if hasattr(step, 'get_feature_names'):
        print(step.get_feature_names())
"""
