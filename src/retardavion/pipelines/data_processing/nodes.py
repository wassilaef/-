import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder


# === 1. Nettoyage des données ===


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage et création des colonnes Scheduled / Operated / Canceled."""
    df["Scheduled"] = pd.to_numeric(df["Nbre mensuel de vols programmés par la Cie sur la relation"], errors="coerce")
    df["Operated"] = pd.to_numeric(df["Nbre mensuel de vols assurés par la Cie sur la relation"], errors="coerce")
    df["Canceled"] = df["Scheduled"] - df["Operated"]

    df = df.dropna(
        subset=[
            "Scheduled",
            "Operated",
            "Canceled",
            "Retard mensuel moyen à l'arrivée des vols exploités par la Cie sur la relation (min)",
        ]
    )
    return df


# === 2. Split X/y et train/test ===


def split_dataset(dataset: pd.DataFrame, test_ratio: float) -> Dict[str, Any]:
    """
    Sépare les données en X_train, X_test, y_train, y_test
    pour la prédiction du retard moyen à l'arrivée.
    """
    X = dataset[["Code Cie", "Code Aero Départ", "Code Aero Destination", "Mois", "Scheduled", "Operated", "Canceled"]]

    y = dataset["Retard mensuel moyen à l'arrivée des vols exploités par la Cie sur la relation (min)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode toutes les colonnes object/bool en entiers avec LabelEncoder."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df
