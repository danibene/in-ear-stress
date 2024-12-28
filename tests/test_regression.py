import numpy as np

from stresspred.data.load_data import AudaceDataLoader
from stresspred.predict.evaluate import get_cv_iterator
from stresspred.predict.train import make_prediction_pipeline


def test_regression(example_data_paths):
    selected_tasks = ["MENTAL", "CPT"]
    selected_signals = ["ECG", "IEML", "ECG-IEMLe"]
    base_cond = "Without baselining"

    selected_features = "All Neurokit2 features"
    rel_values = base_cond == "With baselining"

    est = "Log. Reg."
    sig_name_train = "ECG-IEMLe"
    sig_name_val = "IEML"
    sig_names_train = sig_name_train.split(" + ")
    pred_pipe = make_prediction_pipeline(est=est)
    all_sig_names = np.concatenate(
        (
            np.array(sig_names_train),
            np.array([sig_name_val]),
        )
    )
    selected_signals = list(np.unique(all_sig_names))
    loader = AudaceDataLoader(root=example_data_paths["data_dir"])
    selection_dict = {
        "Task": selected_tasks,
        "Signal": selected_signals,
    }
    in_data = loader.get_ibi_df()
    out = loader.get_split_pred_df(
        selection_dict=selection_dict,
        selected_features=selected_features,
        load_from_file=False,
        save_file=False,
        rel_values=rel_values,
        in_data=in_data,
    )
    X = out["X"]
    y = out["y"]
    sub = out["sub"]
    signal = out["signal"]
    outer_cv, _ = get_cv_iterator(
        sub,
        n_outer_splits=5,
        n_inner_splits=4,
        train_bool=np.array(
            [
                (True if np.any(sig == np.array(sig_names_train)) else False)
                for sig in signal
            ]
        ),
        val_bool=signal == sig_name_val,
    )

    accuracy_scores = []
    for train, test in outer_cv:
        X_train = X.iloc[train]
        y_train = y[train]

        pred_pipe.fit(X_train, y_train)
        X_test = X.iloc[test]
        y_test = y[test]
        accuracy_scores.append(pred_pipe.score(X_test, y_test))

    assert np.mean(accuracy_scores) == 87
