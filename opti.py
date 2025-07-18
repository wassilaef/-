import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# 1. Chargement du vrai dataset
df = pd.read_csv('/Users/wassilaelfarh/Desktop/M2/S2/orch ml/retardavion/data/avion_par_compagnie.csv',sep=None, engine='python')
print("=== Colonnes disponibles dans primary.csv ===")
print(df.columns.tolist())  # ⬅️ À lancer une fois pour vérifier

# 2. Utilisez exactement les noms trouvés ci‑dessus
df['Scheduled'] = pd.to_numeric(
    df['Nbre mensuel de vols programmés par la Cie sur la relation'],
    errors='coerce'
)
df['Operated'] = pd.to_numeric(
    df['Nbre mensuel de vols assurés par la Cie sur la relation'],
    errors='coerce'
)
df['Canceled'] = df['Scheduled'] - df['Operated']

df = df.dropna(subset=[
    'Scheduled', 'Operated', 'Canceled',
    'Retard mensuel moyen à l\'arrivée des vols exploités par la Cie sur la relation (min)'
])

# 3. Définition de X et y
X = df[['Code Cie', 'Code Aero Départ', 'Code Aero Destination', 'Mois',
        'Scheduled', 'Operated', 'Canceled']]
y = df['Retard mensuel moyen à l\'arrivée des vols exploités par la Cie sur la relation (min)'].values

# 4. Pipeline et GridSearch
cat_cols = ['Code Cie', 'Code Aero Départ', 'Code Aero Destination']
num_cols = ['Mois', 'Scheduled', 'Operated', 'Canceled']

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', categorical_transformer, cat_cols),
    ('num', numeric_transformer, num_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        objective='reg:squarederror', random_state=42
    ))
])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 6],
    'regressor__learning_rate': [0.01, 0.1],
    'regressor__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    pipeline, param_grid,
    scoring='neg_root_mean_squared_error',
    cv=3, verbose=2, n_jobs=-1
)

# 5. Entraînement et résultats
grid_search.fit(X, y)
print("✅ Meilleurs hyperparamètres :", grid_search.best_params_)
print(f"✅ RMSE moyen (CV) : {-grid_search.best_score_:.2f}")
