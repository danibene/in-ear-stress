import numpy as np
from sklearn.model_selection import KFold


def get_cv_iterator(
    sub_data,
    num_trials=10,
    n_outer_splits=5,
    n_inner_splits=5,
    train_bool=None,
    val_bool=None,
    inner_val_use_train_bool=True,
    seed=0,
):
    """Repeated subject-wise cross validation"""
    if train_bool is None:
        train_bool = np.ones_like(sub_data).astype(bool)
    if val_bool is None:
        val_bool = np.ones_like(sub_data).astype(bool)
    outer_cv_iterator = []
    inner_cv_iterator = []
    for rand_trial in np.arange(num_trials) + seed:
        rng = np.random.RandomState(rand_trial)
        outer_kf = KFold(n_splits=n_outer_splits, random_state=rng, shuffle=True)
        unique_subs = np.unique(sub_data)
        for outer_train_sub_index, test_sub_index in outer_kf.split(unique_subs):
            outer_train_subs = unique_subs[outer_train_sub_index]
            test_subs = unique_subs[test_sub_index]
            outer_train_index = [
                i
                for i in range(len(sub_data))
                if sub_data[i] in outer_train_subs and train_bool[i]
            ]
            test_index = [
                i
                for i in range(len(sub_data))
                if sub_data[i] in test_subs and val_bool[i]
            ]
            outer_cv_iterator.append((outer_train_index, test_index))
            inner_kf = KFold(n_splits=n_inner_splits, random_state=rng, shuffle=True)
            inner_fold_count = 0
            inner_cv = []
            for inner_train_sub_index, validation_sub_index in inner_kf.split(
                outer_train_subs
            ):
                inner_fold_count += 1
                inner_train_subs = outer_train_subs[inner_train_sub_index]
                validation_subs = outer_train_subs[validation_sub_index]
                inner_train_index = [
                    i
                    for i in range(len(sub_data))
                    if sub_data[i] in inner_train_subs and train_bool[i]
                ]
                if inner_val_use_train_bool:
                    inner_validation_index = [
                        i
                        for i in range(len(sub_data))
                        if sub_data[i] in validation_subs and train_bool[i]
                    ]
                else:
                    inner_validation_index = [
                        i
                        for i in range(len(sub_data))
                        if sub_data[i] in validation_subs and val_bool[i]
                    ]
                inner_cv.append((inner_train_index, inner_validation_index))
            inner_cv_iterator.append(inner_cv)
    return outer_cv_iterator, inner_cv_iterator