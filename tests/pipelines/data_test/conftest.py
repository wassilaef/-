import pytest
import pandas as pd

@pytest.fixture(scope="module")
def project_id():
    # non utilisé ici puisque tu n'as pas de bucket GCS
    return "636759540962"

@pytest.fixture(scope="module")
def primary_folder():
    # chemin vers ton CSV local
    return "data/03_primary/primary.csv"

@pytest.fixture(scope="module")
def dataset_not_encoded(primary_folder):
    # on charge le CSV « brut »
    return pd.read_csv(primary_folder)

@pytest.fixture(scope="module")
def test_ratio():
    return 0.2
