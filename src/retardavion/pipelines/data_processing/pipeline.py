from kedro.pipeline import Pipeline, node
from .nodes import preprocess_data, encode_features, split_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                preprocess_data,
                inputs=["primary", "params:target_column"],
                outputs="dataset",
                name="preprocess_node",
            ),
            node(
                encode_features,
                inputs="dataset",
                outputs="encoded_dataset",
                name="encode_node",
            ),
            node(
                split_dataset,
                inputs=["encoded_dataset", "params:test_ratio", "params:target_column"],  # <—
                outputs=dict(
                    X_train="X_train",
                    X_test="X_test",
                    y_train="y_train",
                    y_test="y_test",
                ),
                name="split_node",
            ),
        ]
    )
