"""Inpatient Row Resampling

Methods for handling row resampling"""

from typing import List, Tuple

import numpy as np
import pandas as pd


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
    """

    def __init__(self, model_run) -> None:
        self._model_run = model_run

        self._row_counts = self._model_run.model.baseline_counts.copy()
        # initialise step counts
        self.step_counts = model_run.step_counts

        self._baseline_counts = (
            self._model_run.model.data_mask * self._model_run.model.baseline_counts
        )

        self.step_counts[("baseline", "-")] = self._baseline_counts.sum(axis=1)

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
    def birth_factors(self):
        """get the birth factors for the model"""
        return self._model_run.model.birth_factors

    @property
    def hsa(self):
        """get the health status adjustment GAMs for the model"""
        return self._model_run.model.hsa

    @property
    def strategies(self):
        """get the activty avoidance strategies for the model"""
        return self._model_run.model.strategies["activity_avoidance"]

    @property
    def data(self):
        """get the current model runs data"""
        return self._model_run.data

    def _update(self, factor: pd.Series, cols: List[str]):
        step = factor.name

        factor = (
            self.data.merge(factor, how="left", left_on=cols, right_index=True)[step]
            .fillna(1)
            .to_numpy()
        )

        self._row_counts *= factor

        self.step_counts[(step, "-")] = ((factor - 1) * self._baseline_counts).sum(
            axis=1
        )
        return self

    def _update_rn(self, factor: pd.Series, group: str):
        factor = self.data["rn"].map(factor).fillna(1).to_numpy()
        self._row_counts *= factor

        self.step_counts[("activity_avoidance", group)] = (
            (factor - 1) * self._baseline_counts
        ).sum(axis=1)
        return self

    def demographic_adjustment(self):
        """perform the demograhic adjustment"""
        year = str(self.run_params["year"])
        variant = self.run_params["variant"]

        factor = self.demog_factors.loc[(variant, slice(None), slice(None))][
            year
        ].rename("demographic_adjustment")
        return self._update(factor, ["age", "sex"])

    def birth_adjustment(self):
        """perform the birth adjustment"""
        year = str(self.run_params["year"])
        variant = self.run_params["variant"]

        b_factor = self.birth_factors.loc[([variant], slice(None), slice(None))][year]
        d_factor = self.demog_factors.loc[b_factor.index][year]

        factor = b_factor / d_factor

        factor = pd.Series(
            factor.values,
            name="birth_adjustment",
            index=pd.MultiIndex.from_tuples(
                [("maternity", a, s) for _, a, s in factor.index.values],
                names=["group", "age", "sex"],
            ),
        )

        return self._update(factor, ["group", "age", "sex"])

    def health_status_adjustment(self):
        """perform the health status adjustment"""
        return self._update(
            self.hsa.run(self.run_params),
            ["hsagrp", "sex", "age"],
        )

    def covid_adjustment(self):
        """perform the covid adjustment"""
        params = self.run_params["covid_adjustment"][self._activity_type]
        factor = pd.Series(params, name="covid_adjustment")
        return self._update(factor, ["group"])

    def expat_adjustment(self):
        """perform the expatriation adjustment"""
        params = {
            k: v
            for k, v in self.run_params["expat"][self._activity_type].items()
            if v  # remove empty values from the dictionary
        }
        if not params:
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
            if v  # remove empty values from the dictionary
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
            {
                k: pd.Series(v, name="baseline_adjustment", dtype="float64")
                for k, v in params.items()
            }
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

        factor = pd.Series(params)

        # update the index to include "True" for the is_wla field
        factor.index = pd.MultiIndex.from_tuples([(True, i) for i in factor.index])
        factor.name = "waiting_list_adjustment"

        return self._update(factor, ["is_wla", "tretspef"])

    def non_demographic_adjustment(self):
        """perform the non-demographic adjustment"""

        if not (
            params := self.run_params["non-demographic_adjustment"][self._activity_type]
        ):
            return self

        year_exponent = self.run_params["year"] - self.params["start_year"]
        factor = pd.Series(params).rename("non-demographic_adjustment") ** year_exponent
        return self._update(factor, ["group"])

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
            self._update_rn(strategies_grouped[k, slice(None)], k)

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
        # note: simple division will cause an issue in the case that there is
        #   no change, i.e. sum_est == sum_before. this ensures that if that is
        #   the case, then we get a value of 1 for the slack
        slack = 1 + np.divide(
            sum_after - sum_est,
            sum_est - sum_before,
            out=np.zeros_like(sum_before),
            where=sum_est != sum_before,
        )
        for k in self.step_counts.keys():
            if k != ("baseline", "-"):
                self.step_counts[k] *= slack
