from kedro.pipeline import Pipeline, node
from .nodes import auto_ml


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=auto_ml,
                inputs=["X_train", "y_train", "params:automl_max_evals"],
                outputs=["trained_model", "best_hypers"],
                name="auto_ml_node",
            ),
        ]
    )
