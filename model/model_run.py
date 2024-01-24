"""Model Run

Provides a simple class which holds all of the data required for a model run
"""
import numpy as np

from model.activity_avoidance import ActivityAvoidance


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
        self.data = model.data.copy()

        # run the model
        self._run()

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

    def _run(self):
        if self.model_run == -1:
            return

        (
            ActivityAvoidance(self)
            .demographic_adjustment()
            .birth_adjustment()
            .health_status_adjustment()
            .covid_adjustment()
            .expat_adjustment()
            .repat_adjustment()
            .waiting_list_adjustment()
            .baseline_adjustment()
            .non_demographic_adjustment()
            .activity_avoidance()
            # call apply_resampling last, as this is what actually alters the data
            .apply_resampling()
        )

        self.model.efficiencies(self)

    def get_aggregate_results(self) -> dict:
        """Aggregate the model results

        Can also be used to aggregate the baseline data by passing in the raw data

        :param model_results: a DataFrame containing the results of a model iteration
        :type model_results: pandas.DataFrame
        :param model_run: the current model run
        :type model_run: int

        :returns: a dictionary containing the different aggregations of this data
        :rtype: dict
        """
        # pylint: disable=assignment-from-no-return
        agg, aggregates = self.model.aggregate(self)
        step_counts = self.get_step_counts()

        return {
            **agg(),
            **agg(["sex", "age_group"]),
            **agg(["age"]),
            **aggregates,
            **step_counts,
        }

    def get_step_counts(self):
        """get the step counts of a model run"""
        if not self.step_counts:
            return {}

        return {"step_counts": self.model.convert_step_counts(self.step_counts)}

    def get_model_results(self):
        """get the model results of a model run"""
        return self.data.reset_index(drop=True).drop(columns=["hsagrp"])
