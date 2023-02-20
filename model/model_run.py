"""Model Run

Provides a simple class which holds all of the data required for a model run
"""
import numpy as np


class ModelRun:
    def __init__(self, model, model_run):
        self._model = model

        self.params = model.params
        self.model_run = model_run
        # if model_run == -1, then use model_run = 0 for run params
        self.run_params = model._get_run_params(max(0, model_run))
        self.rng = np.random.default_rng(self.run_params["seed"])
        self.data = model.data

        self._patch_run_params()
        
        self.step_counts = {}

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

        rp["activity_avoidance"] = {
            "ip": rp["inpatient_factors"]["admission_avoidance"]
        }

        rp["activity_avoidance"]["aae"] = {
            f"{k0}_{k1}": v1
            for k0, v0 in rp["aae_factors"].items()
            for k1, v1 in v0.items()
        }

        rp["activity_avoidance"]["op"] = {
            f"{k0}_{k1}": v1
            for k0, v0 in rp["outpatient_factors"].items()
            for k1, v1 in v0.items()
            if k0 != "convert_to_tele"
        }

        rp["efficiencies"] = {
            "ip": rp["inpatient_factors"]["los_reduction"],
            "op": {
                f"convert_to_tele_{k1}": v1
                for k1, v1 in rp["outpatient_factors"]["convert_to_tele"].items()
            },
        }

    def get_step_counts(self):
        return self._model._get_step_counts_dataframe(self.step_counts)

    def get_model_results(self):
        return self.data.reset_index(drop=True).drop(columns=["hsagrp"])