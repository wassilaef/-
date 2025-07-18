# src/retardavion/pipelines/data_processing/nodes.py
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(
    df: pd.DataFrame,
    target_column: str
) -> pd.DataFrame:
    """
    Nettoyage et création des colonnes Scheduled / Operated / Canceled
    puis on droppe les lignes où l'output ou les features sont NaN.
    """
    df = df.copy()
    df["Scheduled"] = pd.to_numeric(
        df["Nbre mensuel de vols programmés par la Cie sur la relation"],
        errors="coerce"
    )
    df["Operated"] = pd.to_numeric(
        df["Nbre mensuel de vols assurés par la Cie sur la relation"],
        errors="coerce"
    )
    df["Canceled"] = df["Scheduled"] - df["Operated"]

    # on droppe aussi les lignes où la colonne cible est NaN
    df = df.dropna(subset=[
        "Scheduled", "Operated", "Canceled", target_column
    ])
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def split_dataset(
    dataset: pd.DataFrame,
    test_ratio: float,
    target_column: str
) -> Dict[str, Any]:
    """
    Sépare les données encodées en X_train, X_test, y_train, y_test
    en retirant la colonne cible.
    """
    # X = tout sauf la colonne cible
    X = dataset.drop(columns=[target_column])
    # y = la colonne cible
    y = dataset[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42
    )
    return dict(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )