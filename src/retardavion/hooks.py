import os
import mlflow
from kedro.framework.hooks import hook_impl


class MlflowHook:
    @hook_impl
    def before_pipeline_run(self, run_params):
        # 1. on configure l'URI de tracking (si ce n'est pas déjà fait ailleurs)
        project_root = os.getcwd()
        mlflow.set_tracking_uri(f"file://{project_root}/mlruns")
        # 2. on définit l'expérience
        mlflow.set_experiment("retard_avion_experiment")
