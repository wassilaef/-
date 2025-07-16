from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data, split_dataset, encode_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs="primary",
                outputs="dataset",
                name="clean_node",
            ),
            node(
                func=encode_features,
                inputs="dataset",
                outputs="encoded_dataset",
                name="encode_node",
            ),
            node(
                func=split_dataset,
                inputs=["encoded_dataset", "params:test_ratio"],
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
