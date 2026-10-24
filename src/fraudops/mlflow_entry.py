"""Models-from-code entry point: MLflow executes this file instead of unpickling a Python object."""

import mlflow

from fraudops.mlflow_model import ClaimsTriageModel

mlflow.models.set_model(ClaimsTriageModel())
