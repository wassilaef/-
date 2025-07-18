import pytest
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog, MemoryDataSet

from src.retardavion.pipelines.data_processing.pipeline import create_pipeline

def test_pipeline(dataset_not_encoded, test_ratio):
    # On construit un DataCatalog de test
    catalog = DataCatalog({
        "primary": MemoryDataSet(dataset_not_encoded),
        "params:test_ratio": MemoryDataSet(test_ratio)
    })

    # On exécute le pipeline
    runner = SequentialRunner()
    pipeline = create_pipeline()
    output = runner.run(pipeline, catalog)

    # Vérification finale
    assert output["X_train"].shape[0] == output["y_train"].shape[0]
    assert output["X_test"].shape[0] == output["y_test"].shape[0]
