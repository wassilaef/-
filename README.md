# retardavion
Prédiction du retard mensuel moyen des vols
Chaîne complète Kedro + MLflow : depuis le nettoyage des données jusqu’au versioning du modèle.
📂 Structure du projet

retardavion/
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   └── parameters.yml
│   └── dev/  (configs spécifiques)
├── data/
│   ├── 03_primary/       # CSV brut
│   └── 06_models/        # Sorties des pipelines
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── logs/                 # Fichiers de logs
├── src/retardavion/
│   ├── pipelines/
│   │   ├── data_processing/
│   │   │   ├── nodes.py
│   │   │   └── pipeline.py
│   │   └── data_training/
│   │       ├── nodes.py
│   │       └── pipeline.py
│   ├── pipeline_registry.py
│   └── __init__.py
├── README.md
└── pyproject.toml / requirements.txt
🚀 Installation

Cloner le dépôt
git clone <url-du-repo>
cd retardavion
Créer et activer l’environnement
python3 -m venv venv
source venv/bin/activate
Installer les dépendances
pip install -r requirements.txt
⚙️ Configuration

Catalog (conf/base/catalog.yml)
Déclare les jeux de données :
primary:
  type: pandas.CSVDataSet
  filepath: data/03_primary/primary.csv
X_train:
  type: pandas.CSVDataSet
  filepath: data/06_models/X_train.csv
# … et X_test, y_train, y_test
Paramètres (conf/base/parameters.yml)
test_ratio: 0.2
target_column: "Retard mensuel moyen à l'arrivée des vols exploités par la Cie sur la relation (min)"

# MLflow
automl_max_evals: 100
log_to_mlflow: true
experiment_id: "retard_avion_experiment"
mlflow_server_uri: "http://127.0.0.1:5000"
🔄 Pipelines Kedro

1. Data Processing
Nettoyage, encodage, split train/test.
kedro run --pipeline data_processing
preprocess_data → data/03_primary/primary.csv → DataFrame nettoyé
encode_features → encode toutes les colonnes object
split_dataset → génère X_train.csv, X_test.csv, y_train.csv, y_test.csv
2. Data Training
Recherche d’hyper‑paramètres, entraînement final, log vers MLflow + Model Registry.
kedro run --pipeline data_training
optimize_hyp (Hyperopt + RepeatedKFold) → meilleurs params
train_model → entraînement LGBMRegressor
auto_ml
configure MLflow
log : hyper‑params, RMSE train/test, baseline
log modèle et l’enregistre dans le Model Registry (retard_avion_model)
📊 MLflow

Lancer le serveur
mlflow ui --host 127.0.0.1 --port 5000
Explorer
Experiments : “retard_avion_experiment” → métriques, hyper‑params, artefacts
Model Registry : versionner, promouvoir (Staging, Production)
🧪 Tests à venir

Tests pré‑entraînement : cohérence de données, proportions, taille minimale…
Tests post‑entraînement : comportement du modèle vs baseline, cas d’usage.
📈 Résultats & Observations

RMSE train vs test comparé à une baseline (prédiction de la moyenne)
Sur- et sous‑apprentissage observés via différence RMSE train/test
🔜 Prochaines étapes

Mettre en place les tests (pytest + kedro-test)
Déployer la version production du modèle (API FastAPI, Docker)
Surveiller en production (drift, alertes, ré‑entraînement automatique)