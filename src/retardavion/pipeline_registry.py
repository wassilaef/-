from typing import Dict
from kedro.pipeline import Pipeline

# from retardavion.pipelines.data_processing.pipeline import create_pipeline

# def register_pipelines() -> Dict[str, Pipeline]:
#   """Register the project's pipelines."""
#  data_processing_pipeline = create_pipeline()
# return {
#    "__default__": data_processing_pipeline,
#   "data_processing": data_processing_pipeline,
# }

from retardavion.pipelines.data_training.pipeline import create_pipeline as tr_pipeline

from retardavion.pipelines.data_processing.pipeline import create_pipeline as dp_pipeline

from retardavion.pipelines.data_loading.pipeline import create_pipeline as ld_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "__default__": dp_pipeline() + tr_pipeline() + ld_pipeline(),
        "data_processing": dp_pipeline(),
        "data_training": tr_pipeline(),
        "data_loading": ld_pipeline(),
    }
