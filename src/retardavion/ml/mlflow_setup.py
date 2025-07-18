import mlflow
import os

# 1. Spécifier le dossier local de stockage (relatif à la racine du projet)
project_root = os.getcwd()                   # ex. "/Users/.../retardavion"
tracking_path = f"file://{project_root}/mlruns"

mlflow.set_tracking_uri(tracking_path)
