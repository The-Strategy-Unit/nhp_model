"""Model Run

Provides a simple class which holds all of the data required for a model run
"""
import numpy as np


class ModelRun:
    """Model Run

    Holds all of the information for a model run
    """

    def __init__(self, model, model_run):
        self.model = model

        self.model_run = model_run
        # if model_run == -1, then use model_run = 0 for run params
        self.run_params = model._get_run_params(max(0, model_run))
        self.rng = np.random.default_rng(self.run_params["seed"])

        self._patch_run_params()

        self.step_counts = {}

        # data is mutated, so is not a property
        self.data = model.data

    @property
    def params(self):
        """get the models parameters"""
        return self.model.params

    def _patch_run_params(self):
        """Patch Run Parameters

        The run parameters for some items need to be 'patched' so that they include all of the
        fields that are used in that step of the model
        """
        run_params = self.run_params
        for i in ["expat", "repat_local", "repat_nonlocal"]:
            run_params[i]["op"] = {
                g: run_params[i]["op"] for g in ["first", "followup", "procedure"]
            }
            run_params[i]["aae"] = {
                k: {"Other": v} for k, v in run_params[i]["aae"].items()
            }

        run_params["baseline_adjustment"]["aae"] = {
            k: {"Other": v} for k, v in run_params["baseline_adjustment"]["aae"].items()
        }

    def get_step_counts(self):
        """get the step counts of a model run"""
        return self.model.get_step_counts_dataframe(self.step_counts)

    def get_model_results(self):
        """get the model results of a model run"""
        return self.data.reset_index(drop=True).drop(columns=["hsagrp"])
