"""Model Run

Provides a simple class which holds all of the data required for a model run
"""
import numpy as np


class ModelRun:
    def __init__(self, model, model_run):
        self._model = model

        self.params = model.params
        self.model_run = model_run
        self.run_params = model._get_run_params(model_run)
        self.rng = np.random.default_rng(self.run_params["seed"])
        self.data = model.data

        self._patch_run_params()

    def _patch_run_params(self):
        """Patch Run Parameters

        The run parameters for some items need to be 'patched' so that they include all of the
        fields that are used in that step of the model
        """
        rp = self.run_params
        for i in ["expat", "repat_local", "repat_nonlocal"]:
            rp[i]["op"] = {g: rp[i]["op"] for g in ["first", "followup", "procedure"]}
            rp[i]["aae"] = {k: {"Other": v} for k, v in rp[i]["aae"].items()}

        rp["baseline_adjustment"]["aae"] = {
            k: {"Other": v} for k, v in rp["baseline_adjustment"]["aae"].items()
        }
