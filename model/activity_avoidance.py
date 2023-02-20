"""Inpatient Row Resampling

Methods for handling row resampling"""

from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.model_run import ModelRun


class ActivityAvoidance:
    def __init__(
        self,
        model_run: ModelRun,
        counts,
        row_mask=None,
    ):
        self._model_run = model_run
        self.data = model_run.data
        self._activity_type = model_run._model.model_type

        self.params = model_run.params
        self.run_params = model_run.run_params

        self._row_counts = counts.copy()
        if row_mask is None:
            self._row_mask = 1
        else:
            self._row_mask = row_mask.astype(float)
        # initialise step counts
        self._sum = 0.0
        self.step_counts = model_run.step_counts
        self._update_step_counts(("baseline", "-"))

        self._demog_factors = model_run._model._demog_factors
        self._hsa_gams = model_run._model._hsa_gams

        self._strategies = model_run._model._strategies["activity_avoidance"]

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

    def _update_step_counts(self, step: tuple[str, str]) -> None:
        s = (self._row_counts * self._row_mask).sum(axis=1)
        self.step_counts[step] = s - self._sum
        self._sum = s

    def demographic_adjustment(self):
        f = self._demog_factors[self.run_params["variant"]]
        return self._update(f, ["age", "sex"])

    def health_status_adjustment(self):
        # convert the arrays from the life expectancy paramerters into a data frame
        lep = self.params["life_expectancy"]
        ages = np.tile(np.arange(lep["min_age"], lep["max_age"] + 1), 2)
        sexs = np.repeat([1, 2], len(ages) // 2)
        lex = pd.DataFrame({"age": ages, "sex": sexs, "ex": lep["m"] + lep["f"]})
        # adjust the life expectancy column using the health status adjustment parameter
        lex["ex"] *= self.run_params["health_status_adjustment"]
        # caclulate the adjusted age
        lex["adjusted_age"] = lex["age"] - lex["ex"]

        lex.set_index("sex", inplace=True)

        # use the hsa gam's to predict with the adjusted age and the actual age to calculate the
        # health status adjustment factor
        f = pd.concat(
            {
                (h, s): pd.Series(
                    g.predict(lex.loc[int(s), "adjusted_age"])
                    / g.predict(lex.loc[int(s), "age"]),
                    name="health_status_adjustment",
                    index=lex.loc[int(s), "age"],
                )
                for (h, s), g in self._hsa_gams.items()
            }
        )
        return self._update(f, ["hsagrp", "sex", "age"])

    def expat_adjustment(self):
        expat_params = self.run_params["expat"][self._activity_type]
        f = pd.concat({k: pd.Series(v, name="expat") for k, v in expat_params.items()})
        return self._update(f, ["group", "tretspef"])

    def repat_adjustment(self):
        f = pd.concat(
            {
                (is_main_icb, k): pd.Series(v, name="repat")
                for (is_main_icb, repat_type) in [
                    (1, "repat_local"),
                    (0, "repat_nonlocal"),
                ]
                for k, v in self.run_params[repat_type][self._activity_type].items()
            }
        )
        return self._update(f, ["is_main_icb", "group", "tretspef"])

    def baseline_adjustment(self):
        """Create a series of factors for baseline adjustment.

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: a series of floats indicating how often we want to sample that row
        :rtype: pandas.Series
        """
        f = pd.concat(
            {
                k: pd.Series(v, name="baseline_adjustment")
                for k, v in self.run_params["baseline_adjustment"][
                    self._activity_type
                ].items()
            }
        )
        return self._update(f, ["group", "tretspef"])

    def waiting_list_adjustment(self):
        """Create a series of factors for waiting list adjustment.

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame

        :returns: a series of floats indicating how often we want to sample that row
        :rtype: pandas.Series
        """
        tretspef_n = self.data["tretspef"].value_counts()
        wla_param = pd.Series(
            self.run_params["waiting_list_adjustment"][self._activity_type]
        )
        f = (wla_param / tretspef_n).fillna(0) + 1

        # update the index to include "True" for the is_wla field
        f.index = pd.MultiIndex.from_tuples([(True, i) for i in f.index])
        f.name = "waiting_list_adjustment"

        return self._update(f, ["is_wla", "tretspef"])

    def non_demographic_adjustment(self):
        """Create a series of factors for non-demographic adjustment.

        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: a series of floats indicating how often we want to sample that row)
        :rtype: pandas.Series
        """
        f = (
            pd.DataFrame.from_dict(
                self.run_params["non-demographic_adjustment"], orient="index"
            )
            .reset_index()
            .rename(columns={"index": "group"})
            .melt(["group"], var_name="age_group")
            .set_index(["age_group", "group"])["value"]
            .rename("non-demographic_adjustment")
        )
        return self._update(f, ["age_group", "group"])

    def activity_avoidance(self) -> dict:
        rng = self._model_run.rng

        strategies = self._strategies

        p = self.run_params["activity_avoidance"][self._activity_type]

        # if there are no items in params for activity_avoidance then exit
        if p == dict():
            return self

        p = pd.Series(p, name="aaf")

        strategies_grouped = (
            strategies.reset_index()
            .merge(p, left_on="strategy", right_index=True)
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
        # get the random sampling for each row
        rng = self._model_run.rng
        row_samples = rng.poisson(self._row_counts)
        # apply the random sampling, update the data and get the counts
        self._model_run.data, counts = self._model_run._model._apply_resampling(
            row_samples, self.data
        )
        sa = counts.sum(axis=1)
        # after resampling we will have a different amount of rows (sa) from the expectation
        # calculated in the step counts. work out the "slack" we are left with, and adjust all
        # of the steps for this slack.
        # this will result in the sum of the step counts equalling the sum of the rows after
        # resampling
        sb = self.step_counts[("baseline", "-")]
        s = np.array(list(self.step_counts.values())).sum(axis=0)
        # create the slack to add to each step
        slack = 1 + (sa - s) / (s - sb)
        for k in self.step_counts.keys():
            if k != ("baseline", "-"):
                self.step_counts[k] *= slack
        # return the data
        return self._model_run.data, self.step_counts
