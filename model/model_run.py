"""Model Run

Provides a simple class which holds all of the data required for a model run
"""

import numpy as np
import pandas as pd

from model.activity_resampling import ActivityResampling


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
        if self.model_run == 0:
            return

        (
            ActivityResampling(self)
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
        return self.model.aggregate(self), self.get_step_counts()

    def get_step_counts(self):
        """get the step counts of a model run"""
        if not self.step_counts:
            return {}

        def get_counts(data, values):
            return (
                pd.concat(
                    [
                        data,
                        pd.DataFrame(values.transpose(), columns=self.model.measures),
                    ],
                    axis=1,
                )
                .groupby(["sitetret", "pod"], as_index=False)[self.model.measures]
                .sum()
                .melt(["sitetret", "pod"], var_name="measure")
                .assign(activity_type=self.model.model_type)
                .set_index(["activity_type", "sitetret", "pod", "measure"])
                .sort_index()["value"]
            )

        sitetret_pod = self.model.data[["sitetret", "pod"]].reset_index(drop=True)

        step_counts = pd.concat(
            {
                k: get_counts(sitetret_pod, v)
                for k, v in {
                    ("baseline", "-"): self.model.baseline_counts,
                    **self.step_counts,
                }.items()
            }
        )
        step_counts.index.names = (
            ["change_factor", "strategy", "activity_type"]
            + list(sitetret_pod.columns)
            + ["measure"]
        )

        step_counts = self._step_counts_get_type_changes(step_counts)

        return step_counts

    def _step_counts_get_type_changes(self, step_counts):
        step_counts.sort_index(inplace=True)

        return pd.concat(
            [
                step_counts,
                self._step_counts_get_type_change_daycase(step_counts),
                self._step_counts_get_type_change_outpatients(step_counts),
            ]
        )

    def _step_counts_get_type_change_daycase(self, step_counts):
        # get the daycase conversion values
        sc_tc_df = (
            step_counts[
                step_counts.index.isin(
                    ["day_procedures_usually_dc", "day_procedures_occasionally_dc"],
                    level=1,
                )
            ]
            .to_frame()
            .reset_index()
        )
        sc_tc_df["pod"] = "ip_elective_daycase"
        sc_tc_df.loc[sc_tc_df["measure"] == "beddays", "value"] = sc_tc_df.loc[
            sc_tc_df["measure"] == "admissions", "value"
        ].tolist()
        return sc_tc_df.groupby(step_counts.index.names)["value"].sum() * -1

    def _step_counts_get_type_change_outpatients(self, step_counts):
        # get the outpatient conversion values
        sc_tc_df = (
            step_counts[
                step_counts.index.isin(
                    ["day_procedures_usually_op", "day_procedures_occasionally_op"],
                    level=1,
                )
                & (step_counts.index.get_level_values(5) == "admissions")
            ]
            .to_frame()
            .reset_index()
        )

        sc_tc_df["activity_type"] = "op"
        sc_tc_df["pod"] = "op_procedure"
        sc_tc_df["measure"] = "attendances"

        return sc_tc_df.groupby(step_counts.index.names)["value"].sum() * -1

    def get_model_results(self):
        """get the model results of a model run"""
        return self.data.reset_index(drop=True).drop(columns=["hsagrp"])
