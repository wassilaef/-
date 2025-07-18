from kedro.pipeline import Pipeline, node
from .nodes import auto_ml

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=auto_ml,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "params:automl_max_evals",
                    "params:log_to_mlflow",
                    "params:experiment_id",
                    "params:mlflow_server_uri",
                ],
                outputs=["trained_model", "best_hypers"],
                name="auto_ml_node",
            ),
        ]
    )
