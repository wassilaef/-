from kedro.pipeline import Pipeline, node
from .nodes import partition_primary_csv, load_csv_from_bucket


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # 1) Partitionnement de primary.csv en part-*.csv
            node(
                func=partition_primary_csv,
                inputs=["params:primary_csv_path", "params:gcs_primary_folder", "params:partition_chunk_size"],
                outputs=None,
                name="partition_primary_node",
            ),
            # 2) Chargement classique de tous les part-*.csv
            node(
                func=load_csv_from_bucket,
                inputs=["params:gcp_project_id", "params:gcs_primary_folder"],
                outputs="primary",
                name="load_primary_from_parts_node",
            ),
        ]
    )
