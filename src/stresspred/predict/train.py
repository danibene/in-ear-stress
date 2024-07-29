from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_prediction_pipeline(
    scl=StandardScaler(),
    est="LogisticRegression",
    random_state=0,
):

    if isinstance(scl, str):
        if scl == "StandardScaler":
            scaler = StandardScaler()
        elif scl == "None":
            scaler = None
    else:
        scaler = scl

    if isinstance(est, str):
        if est in ["LogisticRegression", "Logistic Regression", "Log. Reg."]:
            from sklearn.linear_model import LogisticRegression

            estimator = LogisticRegression(random_state=random_state, max_iter=1000)
        elif est in ["XGBoost", "XGBClassifier"]:
            from xgboost import XGBClassifier

            estimator = XGBClassifier(random_state=random_state)
    else:
        estimator = est

    p = Pipeline([("scaler", scaler), ("estimator", estimator)])
    return p


def make_search_space(est="LogisticRegression", ref="asare2022"):
    if est in ["LogisticRegression", "Logistic Regression", "Log. Reg."]:
        params = {
            "estimator__C": [0.01, 0.1, 1, 10, 100],
            "estimator__solver": ["newton-cg", "lbfgs", "liblinear", "saga"],
            "estimator__penalty": ["l2"],
        }
    elif est in ["XGBoost", "XGBClassifier"]:
        if ref == "asare2022":
            params = {
                "estimator__learning_rate": [0.05, 0.10, 0.15, 0.20],
                "estimator__max_depth": [3, 4, 5, 6],
                "estimator__gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
                "estimator__min_child_weight": [1, 3, 5, 7],
            }
        else:
            params = {
                "estimator__learning_rate": [0.01, 0.1],
                "estimator__n_estimators": [5, 10, 100],
                "estimator__num_leaves": [5, 16, 31, 62],
            }
    else:
        params = {}
    return params
