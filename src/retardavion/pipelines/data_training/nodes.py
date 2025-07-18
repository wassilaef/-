import mlflow
import mlflow.sklearn
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from hyperopt import Trials, fmin, tpe, hp
from sklearn.model_selection import RepeatedKFold


def optimize_hyp(
    X: np.ndarray,
    y: np.ndarray,
    max_evals: int = 30,
    cv: int = 5,
    repeats: int = 2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Recherche d’hyper‑paramètres avec Hyperopt pour LGBMRegressor."""
    space = {
        "n_estimators": hp.quniform("n_estimators", 50, 200, 10),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
        "max_depth": hp.quniform("max_depth", 3, 6, 1),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 1.0),
        "reg_lambda": hp.uniform("reg_lambda", 0.0, 1.0),
    }

    def objective(params_dict):
        # Convert quniform floats to int
        params = {
            k: int(v) if k in ["n_estimators", "max_depth"] else float(v)
            for k, v in params_dict.items()
        }
        rkf = RepeatedKFold(n_splits=cv, n_repeats=repeats, random_state=random_state)
        scores = []
        for ti, vi in rkf.split(X):
            m = LGBMRegressor(**params, random_state=random_state)
            m.fit(X[ti], y[ti])
            preds = m.predict(X[vi])
            scores.append(-np.sqrt(mean_squared_error(y[vi], preds)))
        return float(np.mean(scores))

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(random_state),
    )
    return {
        "n_estimators": int(best["n_estimators"]),
        "learning_rate": float(best["learning_rate"]),
        "max_depth": int(best["max_depth"]),
        "subsample": float(best["subsample"]),
        "reg_alpha": float(best["reg_alpha"]),
        "reg_lambda": float(best["reg_lambda"]),
    }


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any]
) -> LGBMRegressor:
    """Entraîne un LGBMRegressor avec les params spécifiés."""
    model = LGBMRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    return model


def auto_ml(
    X_train, X_test, y_train, y_test,
    automl_max_evals: int,
    log_to_mlflow: bool,
    experiment_id: str,
    mlflow_server_uri: str,
) -> Tuple[LGBMRegressor, Dict[str, Any]]:
    """
    1) Hyper‑params
    2) Train final + log sur MLflow
    3) Baseline constant
    4) Enregistrement dans le Model Registry
    """
    # 1) Config MLflow
    if log_to_mlflow:
        mlflow.set_tracking_uri(mlflow_server_uri)
        mlflow.set_experiment(experiment_id)

    # 2) Conversion en arrays
    X_tr = X_train.values if hasattr(X_train, "values") else X_train
    y_tr = y_train.values.ravel() if hasattr(y_train, "values") else y_train.ravel()
    X_te = X_test.values if hasattr(X_test, "values") else X_test
    y_te = y_test.values.ravel() if hasattr(y_test, "values") else y_test.ravel()

    # 3) Recherche d’hyper‑params
    best_params = optimize_hyp(X_tr, y_tr, max_evals=automl_max_evals)

    # 4) Entraînement + log
    if log_to_mlflow:
        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # --- modèle LGBM ---
            mlflow.log_params(best_params)
            model = train_model(X_tr, y_tr, best_params)

            # métriques LGBM
            rmse_train = np.sqrt(mean_squared_error(y_tr, model.predict(X_tr)))
            rmse_test  = np.sqrt(mean_squared_error(y_te, model.predict(X_te)))
            mlflow.log_metric("rmse_train", rmse_train)
            mlflow.log_metric("rmse_test",  rmse_test)

            # --- baseline constant ---
            baseline_pred_train = np.full_like(y_tr, y_tr.mean())
            baseline_pred_test  = np.full_like(y_te, y_tr.mean())
            baseline_rmse_train = np.sqrt(mean_squared_error(y_tr, baseline_pred_train))
            baseline_rmse_test  = np.sqrt(mean_squared_error(y_te, baseline_pred_test))
            mlflow.log_metric("baseline_rmse_train", baseline_rmse_train)
            mlflow.log_metric("baseline_rmse_test",  baseline_rmse_test)

            # --- log du modèle comme artifact ---
            mlflow.sklearn.log_model(model, artifact_path="model")

            # 5) Enregistrement dans le Registry
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(
                model_uri,
                name="retard_avion_model"
            )
            print(f"Run URL 👉  {mlflow.get_run(run_id).info.artifact_uri}  | run_id={run_id}")
            print(f"Registered model version: {result.version}")

    else:
        model = train_model(X_tr, y_tr, best_params)

    return model, best_params
