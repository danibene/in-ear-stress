import numpy as np
import pandas as pd
import neurokit2 as nk

from neurokit2.hrv.intervals_utils import _intervals_successive
from stresspred.preprocess.preprocess_utils import peak_time_to_rri


def stress_feat_extract(
    rri,
    rri_time,
    data_format="rri",
    check_successive=True,
    expected_columns=[],
):
    out_dfs = []
    if check_successive:
        rri_info = {"RRI": rri, "RRI_Time": rri_time}
    else:
        rri_info = {"RRI": rri}

    try:
        out_dfs.append(nk.hrv_time(rri_info))
    except:
        pass
    try:
        out_dfs.append(nk.hrv_frequency(rri_info))
    except:
        pass
    try:
        out_dfs.append(nk.hrv_nonlinear(rri_info))
    except:
        pass
    try:
        out_dfs.append(
            my_stress_feat(
                rri=rri, rri_time=rri_time, check_successive=check_successive
            )
        )
    except:
        pass

    if len(out_dfs) < 1:
        out = pd.DataFrame()
    else:
        out = pd.concat(out_dfs, axis=1)

    if len(expected_columns) > len(out.columns):
        cols_to_add = [col for col in expected_columns if col not in out.columns]
        for col in cols_to_add:
            out[col] = np.nan
    return out


def iqmean(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    return np.mean(x[((x >= q1) & (x <= q3))])


def my_stress_feat(rri, rri_time=None, check_successive=True):
    if rri_time is None:
        # Compute the timestamps of the R-R intervals in seconds
        rri_time = np.nancumsum(rri / 1000)
    # Remove NaN R-R intervals, if any
    rri_time = rri_time[~np.isnan(rri)]
    rri = rri[~np.isnan(rri)]

    # Compute the difference between the R-R intervals
    # without checking whether they are successive
    diff_rri = np.diff(rri)

    if check_successive:
        diff_rri = diff_rri[_intervals_successive(rri, rri_time)]
    out = {}
    if len(diff_rri) > 1:
        out["Rt_Median_Sq_SD"] = np.sqrt(np.median(np.square(diff_rri)))
        out["Rt_IQMean_Sq_SD"] = np.sqrt(iqmean(np.square(diff_rri)))
        out["Rt_Prc25_Sq_SD"] = np.sqrt(np.percentile(np.square(diff_rri), 25))

    for window in [2, 3, 4, 5, 10, 15, 20, 30]:
        window_size = window * 1000  # Convert window in s to ms
        if rri_time is None:
            # Compute the timestamps of the R-R intervals in seconds
            rri_time = np.nancumsum(rri / 1000)
        # Convert timestamps to milliseconds and subtract first timestamp
        rri_time_ms_ref_start = (rri_time - rri_time[0]) * 1000
        n_windows = int(np.round(rri_time_ms_ref_start[-1] / window_size))
        if n_windows < 3:
            out["MADM" + str(window)] = np.nan
        else:
            med_rri = []
            for i in range(n_windows):
                start = i * window_size
                start_idx = np.where(rri_time_ms_ref_start >= start)[0][0]
                end_idx = np.where(rri_time_ms_ref_start < start + window_size)[0][-1]
                med_rri.append(np.nanmedian(rri[start_idx:end_idx]))
            out["MADM" + str(window)] = np.nanmedian(
                np.abs(med_rri - np.nanmedian(med_rri))
            )

    for window in [3, 4, 5, 10, 15, 20, 30]:
        window_size = window * 1000  # Convert window in s to ms
        if rri_time is None:
            # Compute the timestamps of the R-R intervals in seconds
            rri_time = np.nancumsum(rri / 1000)
        # Convert timestamps to milliseconds and subtract first timestamp
        rri_time_ms_ref_start = (rri_time - rri_time[0]) * 1000
        n_windows = int(np.round(rri_time_ms_ref_start[-1] / window_size))
        if n_windows < 3:
            out["MMAD" + str(window)] = np.nan
        else:
            mad_rri = []
            for i in range(n_windows):
                start = i * window_size
                start_idx = np.where(rri_time_ms_ref_start >= start)[0][0]
                end_idx = np.where(rri_time_ms_ref_start < start + window_size)[0][-1]
                sel_rri = rri[start_idx:end_idx]
                mad_rri.append(np.nanmedian(np.abs(sel_rri - np.nanmedian(sel_rri))))
            out["MMAD" + str(window)] = np.nanmedian(mad_rri)

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("MY_")
    return out


def get_expected_columns_hrv(func=stress_feat_extract, duration=180):
    ecg = nk.ecg_simulate(duration=duration)
    signals, info = nk.ecg_process(ecg)
    ecg = nk.ecg_simulate(duration=duration)
    _, info = nk.ecg_process(ecg)
    rri, rri_time = peak_time_to_rri(info["ECG_R_Peaks"] / info["sampling_rate"])
    out = func(rri, rri_time)
    return list(out.columns)


def get_selected_features_in_set(set_label="All Neurokit2 features"):
    if set_label == "All Neurokit2 features":
        selected_features = [
            "HRV_MeanNN",
            "HRV_SDNN",
            "HRV_SDANN1",
            "HRV_SDNNI1",
            "HRV_RMSSD",
            "HRV_SDSD",
            "HRV_CVNN",
            "HRV_CVSD",
            "HRV_MedianNN",
            "HRV_MadNN",
            "HRV_MCVNN",
            "HRV_IQRNN",
            "HRV_Prc20NN",
            "HRV_Prc80NN",
            "HRV_pNN50",
            "HRV_pNN20",
            "HRV_MinNN",
            "HRV_MaxNN",
            "HRV_HTI",
            "HRV_TINN",
            "HRV_VLF",
            "HRV_LF",
            "HRV_HF",
            "HRV_VHF",
            "HRV_LFHF",
            "HRV_LFn",
            "HRV_HFn",
            "HRV_LnHF",
            "HRV_SD1",
            "HRV_SD2",
            "HRV_SD1SD2",
            "HRV_S",
            "HRV_CSI",
            "HRV_CVI",
            "HRV_CSI_Modified",
            "HRV_PIP",
            "HRV_IALS",
            "HRV_PSS",
            "HRV_PAS",
            "HRV_GI",
            "HRV_SI",
            "HRV_AI",
            "HRV_PI",
            "HRV_C1d",
            "HRV_C1a",
            "HRV_SD1d",
            "HRV_SD1a",
            "HRV_C2d",
            "HRV_C2a",
            "HRV_SD2d",
            "HRV_SD2a",
            "HRV_Cd",
            "HRV_Ca",
            "HRV_SDNNd",
            "HRV_SDNNa",
            "HRV_DFA_alpha1",
            "HRV_MFDFA_alpha1_Width",
            "HRV_MFDFA_alpha1_Peak",
            "HRV_MFDFA_alpha1_Mean",
            "HRV_MFDFA_alpha1_Max",
            "HRV_MFDFA_alpha1_Delta",
            "HRV_MFDFA_alpha1_Asymmetry",
            "HRV_ApEn",
            "HRV_SampEn",
            "HRV_ShanEn",
            "HRV_FuzzyEn",
            "HRV_MSEn",
            "HRV_CMSEn",
            "HRV_RCMSEn",
            "HRV_CD",
            "HRV_HFD",
            "HRV_KFD",
            "HRV_LZC",
        ]
    elif set_label == "Time and frequency Neurokit2 features":
        selected_features = [
            "HRV_MeanNN",
            "HRV_SDNN",
            "HRV_SDANN1",
            "HRV_SDNNI1",
            "HRV_RMSSD",
            "HRV_SDSD",
            "HRV_CVNN",
            "HRV_CVSD",
            "HRV_MedianNN",
            "HRV_MadNN",
            "HRV_MCVNN",
            "HRV_IQRNN",
            "HRV_Prc20NN",
            "HRV_Prc80NN",
            "HRV_pNN50",
            "HRV_pNN20",
            "HRV_MinNN",
            "HRV_MaxNN",
            "HRV_HTI",
            "HRV_TINN",
            "HRV_MeanNN",
            "HRV_SDNN",
            "HRV_SDANN1",
            "HRV_SDNNI1",
            "HRV_RMSSD",
            "HRV_SDSD",
            "HRV_CVNN",
            "HRV_CVSD",
            "HRV_MedianNN",
            "HRV_MadNN",
            "HRV_MCVNN",
            "HRV_IQRNN",
            "HRV_Prc20NN",
            "HRV_Prc80NN",
            "HRV_pNN50",
            "HRV_pNN20",
            "HRV_MinNN",
            "HRV_MaxNN",
            "HRV_HTI",
            "HRV_TINN",
        ]
    elif set_label in ["Only MedianNN", "MedianNN"]:
        selected_features = ["HRV_MedianNN"]
    elif set_label == "MedianNN + RMSSD":
        selected_features = ["HRV_MedianNN", "HRV_RMSSD"]
    elif " + " in set_label:
        selected_features = set_label.split(" + ")
    else:
        selected_features = [set_label]
    return selected_features
