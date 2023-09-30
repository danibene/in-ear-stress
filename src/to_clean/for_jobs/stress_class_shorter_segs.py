import os
import pathlib
import sys
import numpy as np
import argparse
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from json_tricks import load

parser = argparse.ArgumentParser(
    description="Stress classification with shorter segments"
)
parser.add_argument("--root_out_path", default=None, help="where to save outputs")
parser.add_argument(
    "--root_AudaceData", default=None, help="directory containing AUDACE data"
)
parser.add_argument(
    "--root_P5_StressData", default=None, help="directory containing P5_Stress data"
)


args = parser.parse_args()

if __name__ == "__main__":

    root_out_path = args.root_out_path

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
        P5_StressDataLoader,
        AudaceDataLoader,
        make_prediction_pipeline,
        write_dict_to_json,
        get_cv_iterator,
    )

    results = []
    sum_ibi_tol_perc = 25
    seed = 0
    est = XGBClassifier(random_state=seed)
    rel_values = False
    max_minus_min_ibi_tol = 5
    hop_len = 15
    feat_sets = load("feat_sets.json")
    feat_set_labels = list(feat_sets.keys())
    feat_set_labels.extend(
        ["valid_nk_feats_curr_seg", "valid_nk_feats_seg30", "valid_all_feats_curr_seg"]
    )
    train_loader = AudaceDataLoader()

    for dropna in [False, True]:
        for feat_set_label in feat_set_labels:
            for seg_len in np.arange(1, 13) * 15:

                sum_ibi_tol = seg_len * (sum_ibi_tol_perc / 100)

                possible_feat_sets = feat_sets.copy()

                ibi_df = train_loader.get_ibi_df(load_from_file=True, save_file=False)
                seg_ibi_df = train_loader.resegment_ibi_df(
                    seg_len=seg_len,
                    hop_len=hop_len,
                    in_data=ibi_df.copy(),
                    sum_ibi_tol=sum_ibi_tol,
                    max_minus_min_ibi_tol=max_minus_min_ibi_tol,
                )
                train_loader.set_paths(
                    data_derivatives_dir=pathlib.Path(
                        train_loader.paths["data_derivatives_dir"],
                        "exp",
                        "seg_len" + str(seg_len),
                        "hop_len" + str(hop_len),
                    )
                )
                train_loader.get_ibi_df(
                    load_from_file=False, save_file=True, in_data=seg_ibi_df
                )
                out_train = train_loader.get_split_pred_df(
                    selected_tasks=["MENTAL", "MENTALNOISE", "CPT"],
                    load_from_file=True,
                    save_file=True,
                    rel_values=rel_values,
                    dropna=dropna,
                )
                out_train["X"] = out_train["X"].dropna(axis=1, how="all")

                test_loader = P5_StressDataLoader()
                ibi_df = test_loader.get_ibi_df(load_from_file=True, save_file=False)
                if feat_set_label == "valid_nk_feats_seg30":
                    d = load(
                        str(
                            pathlib.Path(
                                test_loader.paths["data_derivatives_dir"],
                                "exp",
                                "seg_len" + str(30),
                                "hop_len" + str(15),
                                "features.json",
                            )
                        )
                    )
                    possible_feat_sets["valid_nk_feats_seg30"] = d["valid_nk_feats"]
                seg_ibi_df = test_loader.resegment_ibi_df(
                    seg_len=seg_len,
                    hop_len=hop_len,
                    in_data=ibi_df.copy(),
                    sum_ibi_tol=sum_ibi_tol,
                    max_minus_min_ibi_tol=max_minus_min_ibi_tol,
                )
                test_loader.set_paths(
                    data_derivatives_dir=pathlib.Path(
                        test_loader.paths["data_derivatives_dir"],
                        "exp",
                        "seg_len" + str(seg_len),
                        "hop_len" + str(hop_len),
                    )
                )
                test_loader.get_ibi_df(
                    load_from_file=False, save_file=True, in_data=seg_ibi_df
                )
                out_test = test_loader.get_split_pred_df(
                    selected_tasks=["MENTAL", "MENTALNOISE", "CPT"],
                    load_from_file=True,
                    save_file=True,
                    rel_values=rel_values,
                    dropna=dropna,
                )
                out_test["X"] = out_test["X"].dropna(axis=1, how="all")
                pipe_clf = make_prediction_pipeline(est=est)
                intersect_feats = list(
                    set(out_train["X"].columns).intersection(set(out_test["X"].columns))
                )
                nk_feats = [col for col in intersect_feats if "HRV_" in col]
                possible_feat_sets["valid_nk_feats_curr_seg"] = nk_feats
                possible_feat_sets["valid_all_feats_curr_seg"] = intersect_feats
                selected_features = possible_feat_sets[feat_set_label]
                selected_features = [
                    feat for feat in selected_features if feat in intersect_feats
                ]
                outer_cv, _ = get_cv_iterator(
                    out_train["sub"], n_outer_splits=5, n_inner_splits=4,
                )
                cv_results = cross_validate(
                    pipe_clf,
                    out_train["X"].loc[:, selected_features],
                    out_train["y"],
                    cv=outer_cv,
                )
                out_train["accuracy"] = np.mean(cv_results["test_score"])

                pipe_clf.fit(out_train["X"].loc[:, selected_features], out_train["y"])

                out_test["y_pred"] = pipe_clf.predict(
                    out_test["X"].loc[:, selected_features]
                )
                out_test["accuracy"] = pipe_clf.score(
                    out_test["X"].loc[:, selected_features], out_test["y"]
                )
                result = {}
                result["val"] = out_train
                result["test"] = out_test
                result["selected_features"] = selected_features
                result["feat_set_label"] = feat_set_label
                result["seg_len"] = seg_len
                result["dropna"] = dropna
                results.append(result)
                d = {}
                d["seg_len"] = seg_len
                d["valid_nk_feats"] = nk_feats
                write_dict_to_json(
                    d=d,
                    json_path=str(
                        pathlib.Path(
                            test_loader.paths["data_derivatives_dir"], "features.json"
                        )
                    ),
                    rewrite=True,
                )
                write_dict_to_json(
                    results,
                    json_path="stress_class_results_shortersegs.json",
                    rewrite=True,
                )
