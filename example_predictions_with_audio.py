#!/usr/bin/env python

# Imports
import joblib
import json
import neurokit2 as nk
import numpy as np
import onnxruntime as ort
import pandas as pd

from pathlib import Path

import soundfile as sf

from tempbeat.extraction.heartbeat_extraction import hb_extract
from tempbeat.extraction.interval_conversion import peak_time_to_rri


def predict_with_onnx(X, onnx_model_path="default_pred_pipe.onnx"):
    # Load the ONNX model
    sess = ort.InferenceSession(onnx_model_path)

    # Prepare input data for testing
    test_data = np.array(X, dtype=np.float32)
    # Example input data
    input_name = sess.get_inputs()[0].name
    inputs = {input_name: test_data}

    # Run the model to get predictions
    pred = sess.run(None, inputs)

    # Extract the output
    output = pred[0]

    return output


def read_audio_section(filename, start_time = None, stop_time = None):
    track = sf.SoundFile(filename)

    can_seek = track.seekable()  # True
    if not can_seek:
        raise ValueError("Not compatible with seeking")

    sr = track.samplerate
    track_length = len(track) / sr 

    if start_time is None:
        start_frame = 0
    else:
        start_frame = sr * start_time

    if stop_time is None:
        stop_frame = len(track)
    else:
        stop_frame = sr * stop_time

    frames_to_read = stop_frame - start_frame
    track.seek(int(start_frame))
    audio_section = track.read(int(frames_to_read))
    return audio_section, sr


if __name__ == "__main__":
    feature_list_path = "expected_features.json"
    audio, sr = read_audio_section(filename="sub-2021021910_IEML.wav", start_time=0, stop_time=60*3)
    peak_time = hb_extract(audio, sampling_rate=sr, method="temp")
    rri, rri_time = peak_time_to_rri(peak_time)
    X = pd.DataFrame(nk.hrv({"RRI": rri, "RRI_Time": rri_time}))

    # Load features expected by the model
    with open(feature_list_path, "r") as json_file:
        expected_features = json.load(json_file)
    # Fill in any missing features with NaNs
    missing_features = [col for col in expected_features if col not in X.columns]
    for feat in missing_features:
        X[feat] = np.nan
    
    print(predict_with_onnx(X.loc[:, expected_features]))
