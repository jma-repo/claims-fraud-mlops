"""MLflow pyfunc wrapper, so the registered model can also be loaded or served by MLflow itself."""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd

from fraudops.model import ModelBundle


class ClaimsTriageModel(mlflow.pyfunc.PythonModel):
    """Input: the feature columns plus `narrative`. Output: calibrated fraud probability."""

    def load_context(self, context):
        from pathlib import Path

        self.bundle = ModelBundle.load(Path(context.artifacts["bundle"]).parent)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return self.bundle.predict(model_input, model_input["narrative"])
