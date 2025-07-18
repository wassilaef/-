import pandas as pd
import numpy as np
import pytest

from src.retardavion.pipelines.data_processing.nodes import encode_features, split_dataset

# seuils pour tes tests
MIN_SAMPLES = 5000
BALANCE_THRESHOLD = 0.1

def test_encode_features(dataset_not_encoded):
    # on récupère le DataFrame « encodé »
    df = encode_features(dataset_not_encoded)

    # 1) la colonne purchased ne contient que 0 ou 1
    assert df["purchased"].isin([0, 1]).all()

    # 2) toutes les colonnes sont numériques
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df[col])

    # 3) assez d'échantillons
    assert df.shape[0] > MIN_SAMPLES

    # 4) classes équilibrées au moins à BALANCE_THRESHOLD
    proportions = df["purchased"].value_counts() / df.shape[0]
    assert (proportions > BALANCE_THRESHOLD).all()

@pytest.fixture(scope="module")
def dataset_encoded(dataset_not_encoded):
    # on réutilise la fixture précédente
    return encode_features(dataset_not_encoded)

def test_split_dataset(dataset_encoded, test_ratio):
    # split_dataset doit renvoyer (X_train, y_train, X_test, y_test)
    X_train, y_train, X_test, y_test = split_dataset(dataset_encoded, test_ratio)

    # 1) tailles cohérentes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # 2) somme = total
    total = dataset_encoded.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == total

    # 3) scikit-learn fait un ceil pour le test split
    assert np.ceil(total * test_ratio) == X_test.shape[0]
