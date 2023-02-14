"""Inpatient Row Resampling

Methods for handling row resampling"""

import os
import pickle
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd


class RowResampling:
    def __init__(
        self, data_path, data, counts, activity_type, params, run_params, row_mask=None
    ):
        self.data = data
        self._activity_type = activity_type

        self.params = params
        self.run_params = run_params

        self._row_counts = counts.copy()
        if row_mask is None:
            self._row_mask = 1
        else:
            self._row_mask = row_mask.astype(float)
        # initialise step counts
        self._sum = 0.0
        self.step_counts = {}
        self._update_step_counts(("baseline", "-"))

        self._data_path = data_path
        self._load_demog_factors()
        self._load_hsa_gams()

    def _load_demog_factors(self) -> None:
        """Load the demographic factors

        Load the demographic factors csv file and calculate the demographics growth factor for the
        years in the parameters.

        Creates 3 private variables:

          * | `self._demog_factors`: a pandas.DataFrame which has a 1:1 correspondance to the model
            | data and a column for each of the different population projections available
          * | `self._variants`: a list containing the names of the different population projections
            | available
          * `self._probabilities`: a list containing the probability of selecting a given variant
        """
        dfp = self.params["demographic_factors"]
        start_year = str(self.params["start_year"])
        end_year = str(self.params["end_year"])

        merge_cols = ["age", "sex"]

        demog_factors = pd.read_csv(os.path.join(self._data_path, dfp["file"]))
        demog_factors[merge_cols] = demog_factors[merge_cols].astype(int)
        demog_factors["demographic_adjustment"] = (
            demog_factors[end_year] / demog_factors[start_year]
        )
        demog_factors.set_index(merge_cols, inplace=True)

        self._demog_factors = {
            k: v["demographic_adjustment"] for k, v in demog_factors.groupby("variant")
        }

    def _load_hsa_gams(self):
        # load the data that's shared across different model types
        with open(f"{self._data_path}/hsa_gams.pkl", "rb") as hsa_pkl:
            self._hsa_gams = pickle.load(hsa_pkl)

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

    def _update_step_counts(self, step: str) -> None:
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

    def admission_avoidance(
        self, strategies: pd.DataFrame, rng: np.random.Generator
    ) -> dict:
        p = pd.Series(
            self.run_params["inpatient_factors"]["admission_avoidance"], name="aaf"
        )

        strategies_grouped = (
            strategies.reset_index()
            .merge(p, left_on="admission_avoidance_strategy", right_index=True)
            .assign(
                aaf=lambda x: 1 - rng.binomial(1, x["sample_rate"]) * (1 - x["aaf"])
            )
            .set_index(["admission_avoidance_strategy", "rn"])["aaf"]
        )

        for k in strategies_grouped.index.levels[0]:
            self._update(
                strategies_grouped[k, slice(None)].rename(k),
                ["rn"],
                group="admission_avoidance",
            )

        return self

    def apply_resampling(self, rng: np.random.Generator):
        if self._row_counts is None:
            raise Exception("can only apply resampling once")

        fns = {
            "ip": self._apply_resampling_ip,
            "op": self._apply_resampling_op,
            "aae": self._apply_resampling_aae,
        }
        sa = fns[self._activity_type](rng)
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
        # prevent the method from being called a second time
        self._row_counts = None
        # return the data
        return self.data, self.step_counts

    def _apply_resampling_ip(self, rng: np.random.Generator):
        # resample the rows based on the factors that were generated by all of the steps
        row_samples = rng.poisson(self._row_counts[0])
        self.data = self.data.loc[self.data.index.repeat(row_samples)].reset_index(
            drop=True
        )
        # filter out the "oversampled" rows
        mask = (~self.data["bedday_rows"]).astype(float)
        # return the amount of admissions/beddays after resampling
        return np.array([mask.sum(), ((1 + self.data["speldur"]) * mask).sum()])

    def _apply_resampling_op(self, rng: np.random.Generator):
        row_samples = rng.poisson(self._row_counts)
        self.data["attendances"] = row_samples[0]
        self.data["tele_attendances"] = row_samples[1]
        return self.data[["attendances", "tele_attendances"]].sum().to_numpy()

    def _apply_resampling_aae(self, rng: np.random.Generator):
        self.data["arrivals"] = rng.poisson(self._row_counts[0])
        return self.data["arrivals"].sum()
