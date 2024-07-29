import os
import pathlib
import sys
import git
import argparse

parser = argparse.ArgumentParser(description="hb extraction")
parser.add_argument("--root_out_path", default=None, help="where to save outputs")
parser.add_argument("--v", default=0, help="version id")
parser.add_argument(
    "--hb_extract_method", default="temp", help="method for hb extraction"
)
parser.add_argument("--thr_corr_height", default=-3, help="method for hb extraction")


args = parser.parse_args()

if __name__ == "__main__":

    root_out_path = args.root_out_path
    version_id = int(args.v)
    hb_extract_method = args.hb_extract_method
    thr_corr_height = float(args.thr_corr_height)

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
        hb_extract,
        frame_timestamps,
        write_dict_to_json,
        AudaceDataLoader,
    )

    data_format = "DB8k"
    sig_name = "ieml"

    hb_extract_methods = [hb_extract_method]
    print(hb_extract_methods)
    repo = git.Repo(search_parent_directories=True)
    git_hexsha = repo.head.object.hexsha

    hb_extract_algo_kwargs = {
        "max_bpm": 200,
        "min_bpm": 40,
        "denoiser_type": "null",
        "thr_corr_height": thr_corr_height,
        "min_n_confident_peaks": 20,
        "max_time_after_last_peak": 5,
        "clean_method": "own_filt",
        "highcut": 25,
        "relative_peak_height_for_temp_min": -2,
        "relative_peak_height_for_temp_max": 2,
        "temp_time_before_peak": 0.3,
        "temp_time_after_peak": 0.3,
        "fix_corr_peaks_by_height": False,
        "fix_interpl_peaks_by_height": False,
        "relative_rri_min": -2.5,
        "relative_rri_max": 2.5,
        "fixpeaks_by_height_time_boundaries": {
            "before_peak_clean": 0.1,
            "after_peak_clean": 0.1,
            "before_peak_raw": 0.005,
            "after_peak_raw": 0.005,
        },
        "corr_peak_extraction_method": "nk_ecg_process",
        "k_nearest_intervals": 8,
        "n_nan_estimation_method": "round",
        "interpolate_args": {"method": "akima"},
        "use_rri_to_peak_time": True,
        "move_average_rri_window": 3,
        "output_format": "only_final",
        "debug_out_path": None,
    }

    loader = AudaceDataLoader()
    loader.set_paths(data_format=data_format)
    sub_labels = loader.get_dataset_iterator()
    if root_out_path is None:
        root_out_path = pathlib.Path(
            loader.get_paths()["data_derivatives_dir"], "hb_annotations"
        )

    for frame_len in [300, 600, 90, 120]:
        for hb_extract_method in hb_extract_methods:
            print(hb_extract_method)
            version_id += 1
            for sub_label in sub_labels:

                loader = AudaceDataLoader(sub_label=sub_label)
                loader.set_paths(data_format=data_format)
                dataset_label = loader.dataset_label
                sub_id = sub_label
                cond_label = ""

                sig_info = loader.get_sig(data_format=data_format, sig_name=sig_name)

                if "nk" in hb_extract_method:
                    auto_method_acronym = "NK"
                else:
                    auto_method_acronym = hb_extract_method.upper()
                name_peaks_dict = {"ecg": "R_Peak", "ti_ppg": "SP", "ieml": "S1_Peak"}
                label = name_peaks_dict[sig_info["name"]]

                txt_json_base_name = (
                    dataset_label
                    + "_"
                    + sub_label
                    + "-"
                    + sig_info["name"]
                    + "-Ann-Auto-"
                    + auto_method_acronym
                    + "-"
                    + name_peaks_dict[sig_info["name"]]
                    + "_v"
                    + str(version_id)
                )
                txt_file_name = txt_json_base_name + ".txt"
                json_file_name = txt_json_base_name + ".json"
                txt_path = str(
                    pathlib.Path(
                        root_out_path,
                        dataset_label,
                        sub_label,
                        "v" + str(version_id),
                        txt_file_name,
                    )
                )

                json_path = str(
                    pathlib.Path(
                        root_out_path,
                        dataset_label,
                        sub_label,
                        "v" + str(version_id),
                        json_file_name,
                    )
                )

                debug_out_path = str(
                    pathlib.Path(
                        root_out_path,
                        dataset_label,
                        sub_label,
                        "v" + str(version_id),
                        txt_json_base_name + "_debug_out",
                    )
                )

                hb_extract_params = hb_extract_algo_kwargs.copy()
                hb_extract_params["detector_type"] = hb_extract_method
                hb_extract_params["frame_len"] = frame_len
                hb_extract_params["validity"] = True
                hb_extract_params["version"] = version_id
                hb_extract_params["git_hexsha"] = git_hexsha

                write_dict_to_json(hb_extract_params, json_path=json_path)

                if not pathlib.Path(txt_path).is_file():
                    sig_info["peak_time"] = frame_timestamps(
                        func=hb_extract,
                        sig=sig_info["sig"],
                        sig_time=sig_info["time"],
                        frame_len=frame_len,
                        sig_name=sig_info["name"],
                        method=hb_extract_method,
                        hb_extract_algo_kwargs=hb_extract_algo_kwargs,
                        save_file=True,
                        txt_path=txt_path,
                        label=label,
                    )
