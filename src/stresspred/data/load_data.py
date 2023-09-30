import os
import numpy as np
import pandas as pd
import librosa
import pathlib

from warnings import warn

from stresspred.data.paths import code_paths

from stresspred.data.data_utils import (
    identify_header,
    write_sig_to_wav,
    get_frame_start_stop,
)

from stresspred.data.structure_input import (
    create_base_df,
    signal_to_feat_df,
    norm_df,
    prep_df_for_class,
)

from stresspred.preprocess.stress_feat_extract import (
    get_expected_columns_hrv,
    get_selected_features_in_set,
)

from stresspred.preprocess.preprocess_utils import (
    sig_time_to_sampling_rate,
    sampling_rate_to_sig_time,
    get_sig_time_ref_first_samp,
    resample_nonuniform,
    invert_sig,
    detect_invert_ecg,
)


class BioDataLoader:
    def __init__(self, root=None):
        self.root = root
        self.data = pd.DataFrame({"X": [np.nan], "y": [np.nan]})
        self.paths_set = False

    def __len__(self):
        if type(self.data) is tuple:
            return len(self.data[0])
        return len(self.data)

    def __getitem__(self, index):
        if type(self.data) is tuple:
            X, y = self.data.iloc[index]
            return X, y
        X = self.data.iloc[index]
        return X

    def get_sig(
        self,
        sig_name="zephyr_ecg",
        start_time=0.0,
        end_time=None,
        new_sampling_rate=None,
        file_path=None,
        data_format="synchedOriginal",
    ):
        if file_path is None:
            self.get_paths(data_format=data_format)
            sig_path_name = sig_name + "_sig"
            file_path = str(self.paths[sig_path_name])
            if (
                not pathlib.Path(file_path).is_file()
                and data_format != "synchedOriginal"
            ):
                if self.dataset_label.upper() == "AUDACE":
                    from stresspred.data.download_data import (
                        URL_AUDACE_ONLY_DB8K,
                        download_from_url,
                    )

                    out_path = pathlib.Path(
                        pathlib.Path(self.paths["data_derivatives_dir"]),
                        "DB8k" + ".zip",
                    )
                    download_from_url(
                        URL_AUDACE_ONLY_DB8K, out_path=out_path, unzip=True
                    )
                elif self.dataset_label.upper() == "P5_STRESS":
                    from stresspred.data.download_data import (
                        URL_P5_STRESS_ONLY_DB8K,
                        download_from_url,
                    )
                    out_path = pathlib.Path(
                        pathlib.Path(self.paths["data_derivatives_dir"]),
                        "DB8k" + ".zip",
                    )
                    download_from_url(
                        URL_P5_STRESS_ONLY_DB8K, out_path=out_path, unzip=True
                    )
                else:
                    warn(
                        "Warning: File does not exist in this format. Reformatting original."
                    )
                    sig_info = self.get_sig(
                        sig_name=sig_name, new_sampling_rate=new_sampling_rate
                    )
                    self.reformat_sig(sig_info, data_format=data_format)
        else:
            file_path = str(file_path)
        if sig_name == "ieml" or pathlib.Path(file_path).suffix == ".wav":
            if end_time is None:
                sig_ch = librosa.load(file_path, sr=None,
                                      mono=True, offset=start_time)
            else:
                dur = end_time - start_time
                sig_ch = librosa.load(
                    file_path, sr=None, mono=True, offset=start_time, duration=dur,
                )
            sig = sig_ch[0]
            sampling_rate = sig_ch[1]
            sig_time = sampling_rate_to_sig_time(
                sig=sig, sampling_rate=sampling_rate, start_time=start_time
            )
        elif (
            sig_name == "zephyr_ecg"
            or sig_name == "ti_ppg"
            or sig_name == "zephyr_resp"
        ):
            if sig_name == "zephyr_ecg":
                # df = pd.read_csv('matrix.txt',sep=',', header = None,
                # skiprows=1000, chunksize=1000)
                ch_index = 1
            if sig_name == "zephyr_resp":
                ch_index = 1
            if sig_name == "ti_ppg":
                ch_index = 6
            sig_df = pd.read_csv(file_path, header=identify_header(file_path),)
            if len(sig_df.columns) > 1:
                sig_ch = sig_df.iloc[:, ch_index].values.astype(float)
                sig_ch_time, sig_ch = get_sig_time_ref_first_samp(
                    sig_df.iloc[:, 0].values.astype(float), sig=sig_ch
                )
                sampling_rate = sig_time_to_sampling_rate(
                    sig_time=sig_ch_time, method="median"
                )
            else:
                # format with one column is 8k
                sampling_rate = 8000
                sig_ch = sig_df.iloc[:, 0].values
                sig_ch_time = sampling_rate_to_sig_time(
                    sig=sig_ch, sampling_rate=sampling_rate, start_time=start_time
                )
            if start_time is None:
                start_samp = 0
            else:
                start_samp = int(sampling_rate * start_time)
            if end_time is None:
                end_samp = len(sig_ch)
            else:
                end_samp = int(sampling_rate * end_time)
            sig_time = sig_ch_time[start_samp:end_samp]
            sig = sig_ch[start_samp:end_samp]

            if sig_name == "ti_ppg":
                sig = invert_sig(sig)
            elif sig_name == "zephyr_ecg":
                if detect_invert_ecg(sig, sampling_rate=sampling_rate):
                    sig = invert_sig(sig)
            if new_sampling_rate is not None:
                sig, sig_time = resample_nonuniform(
                    sig,
                    sig_time,
                    new_sampling_rate=new_sampling_rate,
                    interpolate_method="linear",
                    use_matlab=True,
                )
                sampling_rate = new_sampling_rate
        return {
            "sig": sig,
            "time": sig_time,
            "sampling_rate": sampling_rate,
            "name": sig_name,
        }

    def reformat_sig(self, sig_info, data_format="DB8k"):
        self.get_paths(data_format=data_format)
        self.paths[sig_info["name"] + "_sig"].resolve().parent.mkdir(
            parents=True, exist_ok=True
        )
        sig_name = sig_info["name"]
        sig_info = write_sig_to_wav(
            sig=sig_info["sig"],
            sig_time=sig_info["time"],
            old_sampling_rate=sig_info["sampling_rate"],
            new_sampling_rate=8000,
            wav_path=str(self.paths[sig_info["name"] + "_sig"]),
        )
        sig_info["name"] = sig_name
        return sig_info

    def get_sub_label(self):
        if isinstance(self.sub_id, (int, np.int16, np.int32, np.int64)):
            if self.sub_id < 10:
                self.sub_label = "P0" + str(self.sub_id)
            else:
                self.sub_label = "P" + str(self.sub_id)
        elif isinstance(self.sub_id, str) and "P" in str(self.sub_id).upper():
            self.sub_label = self.sub_id
            self.sub_id = int(self.sub_label[1:])
        else:
            warn(
                "Warning: Invalid input for participant ID. \
                    Should be an integer or a string containing the letter P"
            )

    def get_paths(self, **kwargs):
        if self.paths_set:
            return self.paths
        else:
            self.set_paths(**kwargs)
            return self.paths

    def set_paths(self, paths=None):
        if paths is None:
            self.paths = {}
        else:
            self.paths = paths
        self.paths_set = True
        return self.paths


class StressBioDataLoader(BioDataLoader):
    def get_ibi_df(self, load_from_file=True, save_file=False, in_data=None):
        if in_data is None and load_from_file:
            self.get_paths()
            if not pathlib.Path(self.paths["ibi_df"]).is_file():
                if self.dataset_label.upper() == "AUDACE":
                    from stresspred.data.download_data import (
                        URL_AUDACE_ONLY_STRESS_CLASS,
                        download_from_url,
                    )
                    download_url = URL_AUDACE_ONLY_STRESS_CLASS

                elif self.dataset_label.upper() == "P5_STRESS":
                    from stresspred.data.download_data import (
                        URL_P5_STRESS_ONLY_STRESS_CLASS,
                        download_from_url,
                    )
                    download_url = URL_P5_STRESS_ONLY_STRESS_CLASS

                out_path = pathlib.Path(
                    pathlib.Path(self.paths["stress_class_data_dir"]).parent,
                    pathlib.Path(
                        self.paths["stress_class_data_dir"]).stem + ".zip",
                )
                download_from_url(
                    download_url, out_path=out_path, unzip=True
                )
            self.ibi_df = pd.read_csv(self.paths["ibi_df"])
        else:
            self.ibi_df = in_data.copy()
        if save_file:
            self.get_paths()
            # if parent path does not exist create it
            pathlib.Path(self.paths["ibi_df"]).resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
            self.ibi_df.to_csv(self.paths["ibi_df"], index=False)
        return self.ibi_df

    def get_feat_dfs(
        self, load_from_file=True, save_file=False, rel_values=False, in_data=None, out_path=None
    ):
        extract_feats = True
        if load_from_file:
            self.get_paths()
            if rel_values:
                if (
                    os.path.exists(self.paths["abs_feat_df"])
                    and os.path.exists(self.paths["base_feat_df"])
                ):
                    self.abs_feat_df = pd.read_csv(self.paths["abs_feat_df"])
                    self.base_feat_df = pd.read_csv(self.paths["base_feat_df"])
                    extract_feats = False
            else:
                if os.path.exists(self.paths["abs_feat_df"]):
                    self.abs_feat_df = pd.read_csv(self.paths["abs_feat_df"])
                    extract_feats = False
        if extract_feats:
            all_sig_ibi_df = self.get_ibi_df(
                load_from_file=load_from_file, save_file=save_file, in_data=in_data)
            # here is where only certain signals could be selected
            signals = all_sig_ibi_df["Signal"].unique()
            # e.g. signals = [1] would be just ECG
            ibi_df = all_sig_ibi_df[all_sig_ibi_df["Signal"].isin(signals)]
            
            if rel_values:
                # create DataFrame with baseline segments for each task
                # for this dataset it is from the rest data
                base_ibi_df = create_base_df(ibi_df)

            # here is where certain tasks could be selected
            # (don't do it before because the rest period before
            # the noise task is used as baseline data)
            tasks = all_sig_ibi_df["Task"].unique()
            # e.g. tasks = [1,3] would be just MENTAL & CPT
            abs_ibi_df = ibi_df[ibi_df["Task"].isin(tasks)]
            expected_columns = get_expected_columns_hrv()
            if rel_values:
                self.base_feat_df = signal_to_feat_df(
                    base_ibi_df, expected_columns=expected_columns
                )
            self.abs_feat_df = signal_to_feat_df(
                abs_ibi_df, expected_columns=expected_columns
            )
            if save_file:
                if out_path is None:
                    # if parent path does not exist create it
                    pathlib.Path(self.paths["abs_feat_df"]).resolve().parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    if rel_values:
                        pathlib.Path(self.paths["base_feat_df"]).resolve().parent.mkdir(
                            parents=True, exist_ok=True
                        )
                    self.abs_feat_df.to_csv(
                        self.paths["abs_feat_df"], index=False)
                    if rel_values:
                        self.base_feat_df.to_csv(
                            self.paths["base_feat_df"], index=False)
                else:
                    if pathlib.Path(out_path).is_dir():
                        abs_feat_df_path = str(
                            pathlib.Path(
                                out_path, pathlib.Path(
                                    self.paths["abs_feat_df"]).name
                            )
                        )
                        if rel_values:
                            base_feat_df_path = str(
                                pathlib.Path(
                                    out_path, pathlib.Path(
                                        self.paths["base_feat_df"]).name
                                )
                            )
                    else:
                        abs_feat_df_path = str(
                            pathlib.Path(
                                pathlib.Path(out_path).parent,
                                pathlib.Path(self.paths["abs_feat_df"]).name,
                            )
                        )
                        if rel_values:
                            base_feat_df_path = str(
                                pathlib.Path(
                                    pathlib.Path(out_path).parent,
                                    pathlib.Path(self.paths["base_feat_df"]).name,
                                )
                            )
                    self.abs_feat_df.to_csv(abs_feat_df_path, index=False)
                    if rel_values:
                        self.base_feat_df.to_csv(base_feat_df_path, index=False)
        if not rel_values:
            self.base_feat_df = None
        return self.abs_feat_df, self.base_feat_df

    def get_rel_df(self, load_from_file=True, save_file=False, in_data=None):
        self.get_feat_dfs(
            load_from_file=load_from_file, save_file=save_file, in_data=in_data
        )
        if os.path.exists(self.paths["rel_feat_df"]) and load_from_file:
            self.rel_feat_df = pd.read_csv(self.paths["rel_feat_df"])
        else:
            self.rel_feat_df = norm_df(self.abs_feat_df, self.base_feat_df)
            if save_file:
                self.rel_feat_df.to_csv(self.paths["rel_feat_df"], index=False)
        return self.rel_feat_df

    def get_pred_df(
        self,
        selected_tasks=None,
        selected_signals=None,
        selection_dict=None,
        load_from_file=True,
        save_file=False,
        rel_values=False,
        in_data=None,
    ):
        # alternatively, extract all features and save in the rel_df structure
        # to later be filtered before the NaN selection
        # since feature extraction takes a while
        if selection_dict is None:
            selection_dict = {"Task": selected_tasks,
                              "Signal": selected_signals}
        self.selected_signals = selection_dict["Signal"]
        if rel_values:
            self.get_rel_df(
                load_from_file=load_from_file, save_file=save_file, in_data=in_data
            )
            feat_df = self.rel_feat_df.copy()
        else:
            self.get_feat_dfs(
                load_from_file=load_from_file, save_file=save_file, rel_values=rel_values, in_data=in_data
            )
            feat_df = self.abs_feat_df.copy()
        for key, value in selection_dict.items():
            if value is not None:
                feat_df = feat_df[feat_df[key].isin(value)]
        self.pred_df = feat_df.reset_index(drop=True)
        self.data = self.pred_df
        return self.pred_df

    def get_split_pred_df(
        self,
        selected_tasks=None,
        selected_signals=None,
        selected_features=None,
        selection_dict=None,
        load_from_file=True,
        save_file=False,
        output_type=dict,
        rel_values=False,
        in_data=None,
        dropna=True,
    ):
        self.get_pred_df(
            selected_tasks=selected_tasks,
            selected_signals=selected_signals,
            selection_dict=selection_dict,
            load_from_file=load_from_file,
            save_file=save_file,
            rel_values=rel_values,
            in_data=in_data,
        )
        if isinstance(selected_features, str):
            selected_features = get_selected_features_in_set(selected_features)
        (
            self.X_data,
            self.y_data,
            self.sub_data,
            self.task_data,
            self.signal_data,
            self.method_data,
            self.subseg_data,
        ) = prep_df_for_class(
            self.pred_df, dropna=dropna, selected_features=selected_features,
        )
        self.data = (self.X_data, self.y_data)
        if output_type == dict:
            return {
                "X": self.X_data,
                "y": self.y_data,
                "sub": self.sub_data,
                "task": self.task_data,
                "signal": self.signal_data,
                "method": self.method_data,
                "subseg_data": self.subseg_data,
            }
        else:
            return self.X_data, self.y_data, self.sub_data, self.task_data

    def resegment_ibi_df(
        self,
        seg_len=180,
        hop_len=180,
        in_data=None,
        sum_ibi_tol=np.inf,
        max_minus_min_ibi_tol=np.inf,
    ):
        if in_data is None:
            self.get_paths()
            all_sig_ibi_df = pd.read_csv(self.paths["ibi_df"])
        else:
            all_sig_ibi_df = in_data.copy()

        # info_keys = ["Signal", "Participant", "Task", "Rest"]

        sub_seg_key = "SubSegIdx"
        rri_key = "Ibi"

        rri_time_key = "IbiTime"

        info_keys = [
            col for col in all_sig_ibi_df.columns if col not in [rri_key, rri_time_key]
        ]

        sig_dict = (
            all_sig_ibi_df.groupby(info_keys)[[rri_key, rri_time_key]]
            .apply(lambda g: list(map(tuple, g.values.tolist())))
            .to_dict()
        )
        new_list_dict = []
        for key in list(sig_dict.keys()):
            rri = np.array(sig_dict[key])[:, 0]
            rri_time = np.array(sig_dict[key])[:, 1]
            frames = get_frame_start_stop(
                rri_time,
                frame_len=seg_len,
                hop_len=hop_len,
                start_index=np.min(rri_time),
            )[0]
            for f in range(len(frames)):
                for r in range(len(rri_time)):
                    if rri_time[r] >= frames[f][0] and rri_time[r] <= frames[f][1]:
                        sub_seg_ind = f + 1
                        new_dict = {info_keys[i]: key[i]
                                    for i in range(len(info_keys))}
                        new_dict[sub_seg_key] = sub_seg_ind
                        new_dict[rri_key] = rri[r]
                        new_dict[rri_time_key] = rri_time[r]
                        new_list_dict.append(new_dict)

        resegmented_ibi_df = pd.DataFrame(new_list_dict)
        if sum_ibi_tol < np.inf:
            resegmented_ibi_df["SumIbi"] = resegmented_ibi_df.groupby(info_keys)[
                "Ibi"
            ].transform("sum")
            resegmented_ibi_df = resegmented_ibi_df[
                np.abs(resegmented_ibi_df["SumIbi"] /
                       1000 - seg_len) < sum_ibi_tol
            ].drop(columns="SumIbi")
        if max_minus_min_ibi_tol < np.inf:
            resegmented_ibi_df["MaxMinusMinIbiTime"] = resegmented_ibi_df.groupby(
                info_keys
            )["IbiTime"].transform("max") - resegmented_ibi_df.groupby(info_keys)[
                "IbiTime"
            ].transform(
                "min"
            )
            resegmented_ibi_df = resegmented_ibi_df[
                np.abs(resegmented_ibi_df["MaxMinusMinIbiTime"] - seg_len)
                < max_minus_min_ibi_tol
            ].drop(columns="MaxMinusMinIbiTime")
        return resegmented_ibi_df

    def get_timestamps_df_for_class(
        self,
        task_names=[
            "mentalNoise",
            "mental",
            "cpt",
            "speechBaseline",
            "speechStressed",
            "noise",
        ],
        uni_rest_duration=180,
        uni_task_duration=180,
        uni_task_start_time_ref_orig_start=-30,
        take_postrest_if_no_prerest=True,
        get_orig_marker_df=False,
    ):

        self.get_paths()
        if self.dataset_label.upper() == "AUDACE":
            marker_df = pd.read_table(
                AudaceDataLoader().get_paths()["marker_df"], header=None
            )
            marker_df.columns = ["start_time", "end_time", "label"]
            records = marker_df.to_dict("records")
            new_recs = []
            for r_ind in range(len(records)):
                rec = records[r_ind]
                if "BASELINEPRE_" in rec["label"] or "_TSK" in rec["label"]:
                    if "BASELINEPRE" in rec["label"]:
                        new_label = "silence"
                    else:
                        new_label = rec["label"].split(
                            "_TSK")[0].lower() + "-task"

                    new_recs.append(
                        {
                            "t": rec["start_time"],
                            "y": "{:02d}".format(r_ind) + "-" + new_label + "-start",
                        }
                    )
                    new_recs.append(
                        {
                            "t": rec["end_time"],
                            "y": "{:02d}".format(r_ind) + "-" + new_label + "-stop",
                        }
                    )

            in_df = pd.DataFrame(new_recs)
        else:
            in_df = pd.read_csv(self.paths["marker_df"])
        if get_orig_marker_df:
            return in_df
        rest_start_index = [
            i for i in range(len(in_df["y"])) if "silence-start" in in_df["y"][i]
        ]
        orig_rest_start_time = np.array(in_df["t"][rest_start_index])
        rest_end_index = [
            i for i in range(len(in_df["y"])) if "silence-stop" in in_df["y"][i]
        ]
        orig_rest_end_time = np.array(in_df["t"][rest_end_index])
        orig_rest_duration = orig_rest_end_time - orig_rest_start_time

        all_uni_rest_start_time = []
        for i in range(len(orig_rest_duration)):
            if orig_rest_duration[i] < uni_rest_duration:
                warn(
                    "Warning: Rest duration was only "
                    + str(orig_rest_duration[i])
                    + ". Skipping."
                )
            else:
                all_uni_rest_start_time.append(
                    orig_rest_start_time[i]
                    + orig_rest_duration[i] / 2
                    - uni_rest_duration / 2
                )
        orig_task_names = np.array(task_names)
        start_suffix = "-task-start"
        task_start_index = []
        uni_task_names = []
        for name in orig_task_names:
            task_start_possible_indices = [
                i
                for i in range(len(in_df["y"]))
                if name + start_suffix in in_df["y"][i]
            ]
            if len(task_start_possible_indices) > 0:
                uni_task_names.append("task_" + name)
                if len(task_start_possible_indices) > 1:
                    warn(
                        "Warning: There were multiple instances of "
                        + name
                        + ". Picking the last instance."
                    )
                    task_start_index.extend(
                        [
                            task_start_possible_indices[
                                np.argmax(
                                    in_df["t"][task_start_possible_indices])
                            ]
                        ]
                    )
                else:
                    task_start_index.extend(task_start_possible_indices)
        uni_task_names = np.array(uni_task_names)
        orig_task_start_time = np.array(in_df["t"][task_start_index])
        uni_task_start_time = orig_task_start_time + uni_task_start_time_ref_orig_start
        uni_task_end_time = uni_task_start_time + uni_task_duration
        uni_rest_names = []
        uni_rest_start_time = []
        uni_rest_end_time = []
        for time in all_uni_rest_start_time:
            indices_after_time = [
                i
                for i in range(len(orig_task_start_time))
                if time + uni_rest_duration < orig_task_start_time[i]
            ]
            if len(indices_after_time) > 0:
                closest_index = np.argmin(
                    orig_task_start_time[indices_after_time])
                uni_rest_names.append(
                    "rest_" + uni_task_names[indices_after_time][closest_index]
                )
                uni_rest_start_time.append(time)
                uni_rest_end_time.append(time + uni_rest_duration)
        if (
            (len(uni_rest_end_time) < len(task_names))
            & (len(uni_rest_end_time) < len(all_uni_rest_start_time))
            & take_postrest_if_no_prerest
        ):
            print(
                "I never implemented taking the rest period after the task, so if that's what you want you should select a different rest period manually"
            )
        all_uni_names = uni_rest_names.copy()
        all_uni_names.extend(uni_task_names)
        timestamp_df_for_class = pd.DataFrame(
            {
                "label": all_uni_names,
                "start_time": np.concatenate(
                    [uni_rest_start_time, uni_task_start_time]
                ),
                "end_time": np.concatenate([uni_rest_end_time, uni_task_end_time]),
            }
        )
        return timestamp_df_for_class


class AudaceDataLoader(StressBioDataLoader):
    def __init__(self, root=None, dataset_label="AUDACE", sub_label="01_F-HG"):
        if root is None:
            root = pathlib.Path(
                code_paths["repo_path"], "local_data", "AUDACE", "data")
        self.root = root
        self.sub_label = sub_label
        self.data = pd.DataFrame({"X": [np.nan], "y": [np.nan]})
        self.dataset_label = dataset_label
        self.paths_set = False

    def __len__(self):
        if type(self.data) is tuple:
            return len(self.data[0])
        return len(self.data)

    def __getitem__(self, index):
        if type(self.data) is tuple:
            X, y = self.data.iloc[index]
            return X, y
        X = self.data.iloc[index]
        return X

    def set_paths(self, paths=None, data_derivatives_dir=None, data_format="DB8k"):
        if paths is None:
            self.paths = {}
            self.paths["data_dir"] = pathlib.Path(self.root)
            if data_derivatives_dir is None:
                self.paths["data_derivatives_dir"] = pathlib.Path(
                    self.paths["data_dir"], "derivatives"
                )
            else:
                self.paths["data_derivatives_dir"] = data_derivatives_dir
            self.paths["stress_class_data_dir"] = pathlib.Path(
                self.paths["data_derivatives_dir"], "stress_class"
            )
            self.paths["ibi_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"],
                "stress_class_arp_ibi_20220210_fs_500.csv",
            )
            self.paths["abs_feat_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"], "abs_feat_df.csv"
            )
            self.paths["base_feat_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"], "base_feat_df.csv"
            )
            self.paths["rel_feat_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"], "rel_feat_df.csv"
            )
            self.paths["marker_df"] = pathlib.Path(
                self.paths["data_derivatives_dir"],
                "uni_mkrs_events",
                self.sub_label + "_UNI_MKR_EVENTS.txt",
            )
            if data_format is not None:
                self.paths["formatted_data_dir"] = pathlib.Path(
                    self.paths["data_derivatives_dir"], data_format
                )

                self.paths["sub_data_dir"] = pathlib.Path(
                    self.paths["formatted_data_dir"], self.sub_label
                )

                self.paths["sig_data_dir"] = self.paths["sub_data_dir"]
                sig_filename_start = self.dataset_label + "-" + self.sub_label + "-"

                sig_filename_end = "-Sig-Raw.wav"
                sig_filename_middles = ["ecg", "ieml", "iemr"]
                sig_filename_keys = ["ecg_sig", "ieml_sig", "iemr_sig"]
                for i in range(len(sig_filename_keys)):
                    self.paths[sig_filename_keys[i]] = pathlib.Path(
                        self.paths["sig_data_dir"],
                        sig_filename_start +
                        sig_filename_middles[i] + sig_filename_end,
                    )
        else:
            self.paths = paths
        self.paths_set = True
        return self.paths

    def get_dataset_iterator(self):
        dataset_iterator = []
        sub_data_dirs = list(pathlib.Path(
            self.root, "derivatives", "DB8k").glob("*"))
        for sub_data_dir in sub_data_dirs:
            sub_label = sub_data_dir.stem
            dataset_iterator.append(sub_label)
        return dataset_iterator


class P5_StressDataLoader(StressBioDataLoader):
    def __init__(self, root=None, sub_id=None, part_id=None, dataset_label="P5_Stress"):
        if root is None:
            data_base_path_rstr = (
                r"Z:\Shared\Documents\RD\RD2\_AudioRD\datasets\Biosignals\CritiasStress"
            )
            root = os.path.join(*data_base_path_rstr.split("\\"))
        self.root = root
        if sub_id is None:
            sub_id = 1
        self.sub_id = sub_id
        self.get_sub_label()
        if part_id is None:
            part_id = 1
        self.part_id = part_id
        self.dataset_label = dataset_label
        self.data = pd.DataFrame({"X": [np.nan], "y": [np.nan]})
        self.paths_set = False

    def set_paths(
        self, data_format="synchedOriginal", paths=None, data_derivatives_dir=None
    ):
        if paths is None:
            self.paths = {}
            self.paths["data_dir"] = pathlib.Path(self.root)
            if data_derivatives_dir is None:
                self.paths["data_derivatives_dir"] = pathlib.Path(
                    self.paths["data_dir"], "data_derivatives"
                )
            else:
                self.paths["data_derivatives_dir"] = data_derivatives_dir
            self.paths["stress_class_data_dir"] = pathlib.Path(
                self.paths["data_derivatives_dir"], "stress_class"
            )
            self.paths["ibi_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"], "ibi_df.csv",
            )
            self.paths["abs_feat_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"], "abs_feat_df.csv"
            )
            self.paths["base_feat_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"], "base_feat_df.csv"
            )
            self.paths["rel_feat_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"], "rel_feat_df.csv"
            )
            if data_format == "synchedOriginal":
                self.paths["formatted_data_dir"] = pathlib.Path(
                    self.root, data_format)

                self.paths["sub_data_dirs"] = list(
                    self.paths["formatted_data_dir"].glob("*prompt*")
                )
                if len(self.paths["sub_data_dirs"]) > 0:
                    self.paths["sub_data_dirs"] = [
                        path
                        for path in self.paths["sub_data_dirs"]
                        if os.path.isdir(str(path))
                    ]

                    self.paths["sub_data_dir"] = self.paths["sub_data_dirs"][
                        self.sub_id - 1
                    ]

                    self.paths["part_dirs"] = list(
                        self.paths["sub_data_dir"].glob("*part*")
                    )

                    if len(self.paths["part_dirs"]) > 0:
                        self.paths["sig_data_dir"] = self.paths["part_dirs"][
                            self.part_id - 1
                        ]
                    else:
                        self.paths["sig_data_dir"] = self.paths["sub_data_dir"]
                    self.paths["ti_ppg_sig"] = pathlib.Path(
                        self.paths["sig_data_dir"], "TI_PPG.csv"
                    )
                    self.paths["zephyr_ecg_sig"] = pathlib.Path(
                        self.paths["sig_data_dir"], "ZephyrECG.csv"
                    )
                    self.paths["zephyr_resp_sig"] = pathlib.Path(
                        self.paths["sig_data_dir"], "ZephyrResp.csv"
                    )
                    self.paths["ieml_sig"] = pathlib.Path(
                        self.paths["sig_data_dir"], "IEM_L-LSL.wav"
                    )
                    if not self.paths["ieml_sig"].is_file():
                        replacement_path = pathlib.Path(
                            self.paths["sig_data_dir"], "IEM_L.wav"
                        )
                        warn(
                            "Warning: "
                            + str(self.paths["ieml_sig"])
                            + " does not exist. \n"
                            + "Loading "
                            + str(replacement_path)
                            + " instead."
                        )
                        self.paths["ieml_sig"] = replacement_path
                    self.paths["marker_df"] = pathlib.Path(
                        self.paths["sig_data_dir"], "mrkrConditions.csv"
                    )
            else:
                self.paths["formatted_data_dir"] = pathlib.Path(
                    self.paths["data_derivatives_dir"], data_format
                )

                self.paths["sub_data_dir"] = pathlib.Path(
                    self.paths["formatted_data_dir"], self.sub_label
                )

                self.paths["sig_data_dir"] = self.paths["sub_data_dir"]
                sig_filename_start = (
                    self.dataset_label
                    + "-"
                    + self.sub_label
                    + "_"
                    + str(self.part_id)
                    + "-"
                )

                sig_filename_end = "-Sig-Raw.wav"
                sig_filename_middles = ["TiPpg", "ZephyrEcg", "Ieml"]
                sig_filename_keys = ["ti_ppg_sig",
                                     "zephyr_ecg_sig", "ieml_sig"]

                for i in range(len(sig_filename_keys)):
                    self.paths[sig_filename_keys[i]] = pathlib.Path(
                        self.paths["sig_data_dir"],
                        sig_filename_start +
                        sig_filename_middles[i] + sig_filename_end,
                    )
        else:
            self.paths = paths
        self.paths_set = True
        return self.paths

    def get_all_sub_part_ids(self):
        self.get_paths(data_format="synchedOriginal")
        sub_id = self.sub_id
        part_id = self.part_id
        all_sub_part_ids = []
        for temp_sub_id in np.arange(1, len(self.paths["sub_data_dirs"]) + 1):
            self.sub_id = temp_sub_id
            self.get_sub_label()
            self.set_paths(data_format="synchedOriginal")
            self.get_paths()
            n_parts = len(self.paths["part_dirs"])
            if n_parts == 0:
                n_parts = 1
            for temp_part_id in np.arange(1, n_parts + 1):
                all_sub_part_ids.append((temp_sub_id, temp_part_id))
        # reset self to defaults/previous value
        self.sub_id = sub_id
        self.part_id = part_id
        self.set_paths()
        self.get_paths()
        return all_sub_part_ids


class P5M5DataLoader(StressBioDataLoader):
    def __init__(
        self, root=None, sub_id=3, cond_label="01_preSilence", dataset_label="P5M5_3"
    ):
        if root is None:
            data_base_path_rstr = (
                r"Z:\Shared\Documents\RD\RD2\_AudioRD\datasets\Biosignals"
            )
            root = os.path.join(*data_base_path_rstr.split("\\"))
        self.root = root
        self.sub_id = sub_id
        self.cond_label = cond_label
        self.dataset_label = dataset_label
        self.data = pd.DataFrame({"X": [np.nan], "y": [np.nan]})
        self.paths_set = False

    def set_paths(self, data_format="8k", data_derivatives_dir=None):
        self.paths = {}
        self.get_sub_label()
        self.paths["data_dir"] = pathlib.Path(
            self.root, self.dataset_label, data_format, self.sub_label, self.cond_label
        )

        self.paths["sub_data_dirs"] = [
            path for path in list(pathlib.Path(self.root).glob("P*")) if path.is_dir()
        ]

        self.paths["ti_ppg_sig"] = pathlib.Path(
            self.paths["data_dir"], "TI_PPG.csv")
        self.paths["zephyr_ecg_sig"] = pathlib.Path(
            self.paths["data_dir"], "ZephyrECG.csv"
        )
        self.paths["ecg_audio_sig"] = pathlib.Path(
            self.paths["data_dir"], "ECG_audio.wav"
        )
        self.paths["ieml_sig"] = pathlib.Path(
            self.paths["data_dir"], "IEM_L.wav")

        self.paths["original_sub_data_dir"] = [
            path
            for path in list(
                pathlib.Path(self.root, self.dataset_label,
                             "original").glob("P*")
            )
            if path.is_dir() and self.sub_label in str(path)
        ][0]

        self.paths["marker_df"] = pathlib.Path(
            self.paths["original_sub_data_dir"], "mrkrConditions.csv"
        )

        self.paths_set = True

        self.paths["data_derivatives_dir"] = data_derivatives_dir
        self.paths["stress_class_data_dir"] = pathlib.Path(
            self.paths["data_derivatives_dir"], "stress_class"
        )
        self.paths["ibi_df"] = pathlib.Path(
            self.paths["stress_class_data_dir"], "ibi_df.csv",
        )
        self.paths["abs_feat_df"] = pathlib.Path(
            self.paths["stress_class_data_dir"], "abs_feat_df.csv"
        )
        self.paths["base_feat_df"] = pathlib.Path(
            self.paths["stress_class_data_dir"], "base_feat_df.csv"
        )
        self.paths["rel_feat_df"] = pathlib.Path(
            self.paths["stress_class_data_dir"], "rel_feat_df.csv"
        )

        return self.paths

    def get_all_sub_part_ids(self):
        self.get_paths()
        all_sub_part_ids = []
        temp_part_id = 1
        for temp_sub_id in np.arange(1, len(self.paths["sub_data_dirs"]) + 1):
            all_sub_part_ids.append((temp_sub_id, temp_part_id))

        return all_sub_part_ids

    def get_dataset_iterator(self, unsegmented=False):
        data_format = "8k"
        dataset_iterator = []
        dataset_dirs = list(pathlib.Path(self.root).glob("P5M5*"))
        for dataset_dir in dataset_dirs:
            dataset_label = dataset_dir.stem
            sub_data_dirs = list(pathlib.Path(
                dataset_dir, data_format).glob("P*"))
            for sub_data_dir in sub_data_dirs:
                sub_label = sub_data_dir.stem
                if unsegmented:
                    cond_data_dirs = [
                        d
                        for d in list(pathlib.Path(sub_data_dir).glob("*"))
                        if d.is_dir() and d.stem == "_unsegmented"
                    ]
                else:
                    cond_data_dirs = [
                        d
                        for d in list(pathlib.Path(sub_data_dir).glob("*"))
                        if d.is_dir() and d.stem != "_unsegmented"
                    ]
                for cond_data_dir in cond_data_dirs:
                    cond_label = cond_data_dir.stem
                    dataset_iterator.append(
                        (dataset_label, sub_label, cond_label))
        return dataset_iterator


class GUDBDataLoader(StressBioDataLoader):
    def __init__(self, root, sub_id=None, part_id=None, dataset_label="GUDB"):
        self.root = root
        if sub_id is None:
            sub_id = 1
        self.sub_id = sub_id
        self.get_sub_label()
        if part_id is None:
            part_id = 1
        self.part_id = part_id
        self.dataset_label = dataset_label
        self.data = pd.DataFrame({"X": [np.nan], "y": [np.nan]})
        self.paths_set = False

    def set_paths(self, paths=None, data_derivatives_dir=None, data_format="DB8k"):
        if paths is None:
            self.paths = {}
            self.paths["data_dir"] = pathlib.Path(self.root)
            if data_derivatives_dir is None:
                self.paths["data_derivatives_dir"] = pathlib.Path(
                    self.paths["data_dir"], "data_derivatives"
                )
            else:
                self.paths["data_derivatives_dir"] = data_derivatives_dir
            self.paths["stress_class_data_dir"] = pathlib.Path(
                self.paths["data_derivatives_dir"], "stress_class"
            )
            self.paths["ibi_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"],
                "ibi_df.csv",
            )
            self.paths["abs_feat_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"], "abs_feat_df.csv"
            )
            self.paths["base_feat_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"], "base_feat_df.csv"
            )
            self.paths["rel_feat_df"] = pathlib.Path(
                self.paths["stress_class_data_dir"], "rel_feat_df.csv"
            )
        else:
            self.paths = paths
        self.paths_set = True
        return self.paths
