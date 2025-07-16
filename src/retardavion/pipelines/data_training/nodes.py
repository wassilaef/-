import numpy as np
from typing import Tuple, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold
from hyperopt import Trials, fmin, tpe, hp
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import accuracy_score


def optimize_hyp(
    X: np.ndarray,
    y: np.ndarray,
    max_evals: int = 50,
    cv: int = 5,
    repeats: int = 2,
    random_state: int = 42,
    metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
) -> Dict[str, Any]:
    """
    Espace de recherche défini en dur avec hp.quniform / hp.uniform
    """
    space = {
        "n_estimators": hp.quniform("n_estimators", 50, 200, 10),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
    }

    def objective(params_dict):
        # Convertir float → int pour les paramètres quniform
        params_dict = {k: int(v) if k in ["n_estimators", "max_depth"] else float(v) for k, v in params_dict.items()}
        rkf = RepeatedKFold(n_splits=cv, n_repeats=repeats, random_state=random_state)
        scores = []
        for ti, vi in rkf.split(X):
            X_tr, X_val = X[ti], X[vi]
            y_tr, y_val = y[ti], y[vi]
            model = LGBMClassifier(**params_dict)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(metric(y_val, preds))
        return -float(np.mean(scores))

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(random_state),
    )
    # Reconvertit les choix quniform
    return {
        "n_estimators": int(best["n_estimators"]),
        "learning_rate": float(best["learning_rate"]),
        "max_depth": int(best["max_depth"]),
        "subsample": float(best["subsample"]),
    }


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any]) -> BaseEstimator:
    """Entraîne un LGBMClassifier avec params."""
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def auto_ml(X_train, y_train, automl_max_evals: int) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    1) Optimize hyper‑params sur automl_max_evals essais
    2) Retrain du modèle final
    """
    # Passe de DataFrame → numpy
    X_np = X_train.values if hasattr(X_train, "values") else X_train
    y_np = y_train.values.ravel()  # .ravel() pour éviter l'avertissement
    best_params = optimize_hyp(X_np, y_np, max_evals=automl_max_evals)
    model = train_model(X_np, y_np, best_params)
    return model, best_params
