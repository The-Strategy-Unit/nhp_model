"""Inpatient Row Resampling

Methods for handling row resampling"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.model_run import ModelRun


class ActivityAvoidance:
    """Activity Avoidance

    Class for handling the activity avoidance methods in the model. The class keeps track of
    the current row counts, which represent the value for the lambda parameter to a random poisson
    when we come to resample the rows, and the step counts (the estimated effect of each step on
    the total number of rows).

    The public methods of this class are intended to each be called either once, or not at all.
    These methods update the row counts by multiplying the current value by the factor generated
    from that method.

    Once all of the methods have been run, finally we need to call the `apply_resampling` method.
    This updates the `model_run` which is passed in at initialisation.

    :param model_run: the model run object, which contains all of the required values to run the
    model.
    :type model_run: ModelRun
    :param counts: the baseline counts we need to keep track of.
    :type counts: npt.ArrayLike
    :param row_mask: an array the length of the data which either contains a value of 1 (if that
    row should be included in counts) or 0 (if that row should now be included in counts). Only
    required by inpatients, where rows are oversampled. Defaults to a singleton value of 1.0.
    :type row_mask: npt.ArrayLike
    """

    def __init__(
        self,
        model_run: ModelRun,
        counts: npt.ArrayLike,
        row_mask: npt.ArrayLike,
    ) -> None:
        self._model_run = model_run

        self._row_counts = counts.copy()
        self._row_mask = row_mask.astype(float)
        # initialise step counts
        self._sum = 0.0
        self.step_counts = model_run.step_counts
        self._update_step_counts(("baseline", "-"))

    @property
    def _activity_type(self):
        return self._model_run.model.model_type

    @property
    def params(self):
        """get the models params"""
        return self._model_run.params

    @property
    def run_params(self):
        """get the current params for the model run"""
        return self._model_run.run_params

    @property
    def demog_factors(self):
        """get the demographic factors for the model"""
        return self._model_run.model.demog_factors

    @property
    def hsa_gams(self):
        """get the health status adjustment GAMs for the model"""
        return self._model_run.model.hsa_gams

    @property
    def strategies(self):
        """get the activty avoidance strategies for the model"""
        return self._model_run.model.strategies["activity_avoidance"]

    @property
    def data(self):
        """get the current model runs data"""
        return self._model_run.data

    def _update(self, factor: pd.Series, cols: List[str], group=None):
        step = factor.name

        self._row_counts *= (
            self.data.merge(factor, how="left", left_on=cols, right_index=True)[step]
            .fillna(1)
            .to_numpy()
        )

        if group is not None:
            step, group = group, step

        self._update_step_counts((step, group or "-"))
        return self

    def _update_step_counts(self, step: Tuple[str, str]) -> None:
        new_sum = (self._row_counts * self._row_mask).sum(axis=1)
        self.step_counts[step] = new_sum - self._sum
        self._sum = new_sum

    def demographic_adjustment(self):
        """perform the demograhic adjustment"""
        factor = self.demog_factors[self.run_params["variant"]]
        return self._update(factor, ["age", "sex"])

    def health_status_adjustment(self):
        """perform the health status adjustment"""
        precomputed_activity_ages = self._model_run.model.hsa_precomputed_activity_ages

        ages = precomputed_activity_ages.index.levels[2]

        adjusted_age = precomputed_activity_ages.index.get_level_values(2)
        life_expectancy = (
            precomputed_activity_ages["life_expectancy"]
            * self.run_params["health_status_adjustment"]
        )

        adjusted_age -= life_expectancy

        # use the hsa gam's to predict with the adjusted age and the actual age to calculate the
        # health status adjustment factor
        factor = pd.concat(
            {
                (h, s): pd.Series(
                    g.predict(adjusted_age.loc[h, int(s), slice(None)]),
                    name="health_status_adjustment",
                    index=ages,
                )
                for (h, s), g in self.hsa_gams.items()
            }
        ).rename_axis(["hsagrp", "sex", "age"])
        factor /= precomputed_activity_ages["activity_age"]

        return self._update(factor, ["hsagrp", "sex", "age"])

    def expat_adjustment(self):
        """perform the expatriation adjustment"""
        if not (params := self.run_params["expat"][self._activity_type]):
            return self

        factor = pd.concat({k: pd.Series(v, name="expat") for k, v in params.items()})
        return self._update(factor, ["group", "tretspef"])

    def repat_adjustment(self):
        """perform the repatriation adjustment"""
        params = {
            (is_main_icb, k): pd.Series(v, name="repat")
            for (is_main_icb, repat_type) in [
                (1, "repat_local"),
                (0, "repat_nonlocal"),
            ]
            for k, v in self.run_params[repat_type][self._activity_type].items()
        }
        if not params:
            return self

        factor = pd.concat(params)
        return self._update(factor, ["is_main_icb", "group", "tretspef"])

    def baseline_adjustment(self):
        """perform the baseline adjustment

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline.
        """
        if not (params := self.run_params["baseline_adjustment"][self._activity_type]):
            return self

        factor = pd.concat(
            {k: pd.Series(v, name="baseline_adjustment") for k, v in params.items()}
        )
        return self._update(factor, ["group", "tretspef"])

    def waiting_list_adjustment(self):
        """perform the waiting list adjustment

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline.
        """
        activity_type = self._activity_type
        if activity_type == "aae":
            return self

        if not (params := self.run_params["waiting_list_adjustment"][activity_type]):
            return self

        tretspef_n = self.data["tretspef"].value_counts()
        wla_param = pd.Series(params)
        factor = (wla_param / tretspef_n).fillna(0) + 1

        # update the index to include "True" for the is_wla field
        factor.index = pd.MultiIndex.from_tuples([(True, i) for i in factor.index])
        factor.name = "waiting_list_adjustment"

        return self._update(factor, ["is_wla", "tretspef"])

    def non_demographic_adjustment(self):
        """perform the non-demographic adjustment"""
        if self._activity_type != "ip":
            return self

        if not (params := self.run_params["non-demographic_adjustment"]):
            return self

        factor = (
            pd.DataFrame.from_dict(params, orient="index")
            .reset_index()
            .rename(columns={"index": "group"})
            .melt(["group"], var_name="age_group")
            .set_index(["age_group", "group"])["value"]
            .rename("non-demographic_adjustment")
        )
        return self._update(factor, ["age_group", "group"])

    def activity_avoidance(self) -> dict:
        """perform the activity avoidance (strategies)"""
        # if there are no items in params for activity_avoidance then exit
        if not (params := self.run_params["activity_avoidance"][self._activity_type]):
            return self

        rng = self._model_run.rng

        strategies = self.strategies

        params = pd.Series(params, name="aaf")

        strategies_grouped = (
            strategies.reset_index()
            .merge(params, left_on="strategy", right_index=True)
            .assign(
                aaf=lambda x: 1 - rng.binomial(1, x["sample_rate"]) * (1 - x["aaf"])
            )
            .set_index(["strategy", "rn"])["aaf"]
        )

        for k in strategies_grouped.index.levels[0]:
            self._update(
                strategies_grouped[k, slice(None)].rename(k),
                ["rn"],
                group="activity_avoidance",
            )

        return self

    def apply_resampling(self):
        """apply the row resampling to the data"""
        # get the random sampling for each row
        rng = self._model_run.rng
        row_samples = rng.poisson(self._row_counts)
        # apply the random sampling, update the data and get the counts
        self._model_run.data, counts = self._model_run.model.apply_resampling(
            row_samples, self.data
        )
        sum_after = counts.sum(axis=1)
        # after resampling we will have a different amount of rows (sa) from the expectation
        # calculated in the step counts. work out the "slack" we are left with, and adjust all
        # of the steps for this slack.
        # this will result in the sum of the step counts equalling the sum of the rows after
        # resampling
        sum_before = self.step_counts[("baseline", "-")]
        sum_est = np.array(list(self.step_counts.values())).sum(axis=0)
        # create the slack to add to each step
        slack = 1 + (sum_after - sum_est) / (sum_est - sum_before)
        for k in self.step_counts.keys():
            if k != ("baseline", "-"):
                self.step_counts[k] *= slack
