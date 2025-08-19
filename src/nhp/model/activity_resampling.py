"""Inpatient Row Resampling.

Methods for handling row resampling
"""

import numpy as np
import pandas as pd


class ActivityResampling:
    """Activity Resampling.

    Class for handling the activity resampling methods in the model. The class keeps track
    of the current row counts, which represent the value for the lambda parameter to a
    random poisson when we come to resample the rows, and the step counts (the estimated
    effect of each step on the total number of rows).

    The public methods of this class are intended to each be called either once, or not
    at all.
    These methods update the row counts by multiplying the current value by the factor
    generated from that method.

    Once all of the methods have been run, finally we need to call the `apply_resampling`
    method.
    This updates the `model_iteration` which is passed in at initialisation.

    :param model_iteration: the model iteration object, which contains all of the required
    values to run the model.
    :type model_iteration: ModelIteration
    """

    def __init__(self, model_iteration) -> None:
        """Initialise ActivityResampling.

        :param model_iteration: the current model iteration we are performing
        :type model_iteration: ModelIteration
        """
        self._model_iteration = model_iteration

        # initialise step counts
        self.factors = []

    @property
    def _baseline_counts(self):
        return self._model_iteration.model.baseline_counts

    @property
    def _activity_type(self):
        return self._model_iteration.model.model_type

    @property
    def params(self):
        """Get the models params."""
        return self._model_iteration.params

    @property
    def run_params(self):
        """Get the current params for the model run."""
        return self._model_iteration.run_params

    @property
    def demog_factors(self):
        """Get the demographic factors for the model."""
        return self._model_iteration.model.demog_factors

    @property
    def birth_factors(self):
        """Get the birth factors for the model."""
        return self._model_iteration.model.birth_factors

    @property
    def hsa(self):
        """Get the health status adjustment GAMs for the model."""
        return self._model_iteration.model.hsa

    @property
    def data(self):
        """Get the current model runs data."""
        return self._model_iteration.data

    def _update(self, factor: pd.Series):
        step = factor.name

        factor = self.data.merge(factor, how="left", left_on=factor.index.names, right_index=True)[
            step
        ].fillna(1)

        self.factors.append(factor)

        return self

    def demographic_adjustment(self):
        """Perform the demograhic adjustment."""
        year = str(self.run_params["year"])
        variant = self.run_params["variant"]

        factor = self.demog_factors.loc[(variant, slice(None), slice(None))][year].rename(
            "demographic_adjustment"
        )

        groups = set(self.data["group"]) - {"maternity"}
        factor: pd.Series = pd.concat({i: factor for i in groups})  # type: ignore
        factor.index.names = ["group", *factor.index.names[1:]]

        return self._update(factor)

    def birth_adjustment(self):
        """Perform the birth adjustment."""
        year = str(self.run_params["year"])
        variant = self.run_params["variant"]

        factor = self.birth_factors.loc[([variant], slice(None), slice(None))][year]

        factor = pd.Series(
            factor.values,
            name="birth_adjustment",
            index=pd.MultiIndex.from_tuples(
                [("maternity", a, s) for _, a, s in factor.index.to_numpy()],
                names=["group", "age", "sex"],
            ),
        )

        return self._update(factor)

    def health_status_adjustment(self):
        """Perform the health status adjustment."""
        if not self.params["health_status_adjustment"]:
            return self

        return self._update(self.hsa.run(self.run_params))

    def inequalities_adjustment(self):
        """Perform the inequalities adjustment."""
        activity_type = self._activity_type

        match activity_type:
            case "op":
                factor_key = "procedure"
            case "ip":
                factor_key = "elective"
            case _:
                return self

        if not (params := self.run_params["inequalities"]):
            return self

        factor: pd.Series = pd.concat(  # type: ignore
            {k: pd.Series(v, name="inequalities", dtype="float64") for k, v in params.items()}
        )

        factor.index.names = ("sushrg_trimmed", "imd_quintile")

        factor.index = factor.index.set_levels(  # type: ignore
            factor.index.levels[1].astype(int),  # type: ignore
            level="imd_quintile",
        )

        factor: pd.Series = pd.concat({factor_key: factor}, names=["group"])  # type: ignore

        return self._update(factor)

    def expat_adjustment(self):
        """Perform the expatriation adjustment."""
        params = {
            k: v
            for k, v in self.run_params["expat"][self._activity_type].items()
            if v  # remove empty values from the dictionary
        }
        if not params:
            return self

        factor: pd.Series = pd.concat(  # type: ignore
            {k: pd.Series(v, name="expat") for k, v in params.items()}
        )
        factor.index.names = ["group", "tretspef_grouped"]
        return self._update(factor)

    def repat_adjustment(self):
        """Perform the repatriation adjustment."""
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

        factor: pd.Series = pd.concat(params)  # type: ignore
        factor.index.names = ["is_main_icb", "group", "tretspef_grouped"]
        return self._update(factor)

    def baseline_adjustment(self):
        """Perform the baseline adjustment.

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline.
        """
        if not (params := self.run_params["baseline_adjustment"][self._activity_type]):
            return self

        factor: pd.Series = pd.concat(  # type: ignore
            {
                k: pd.Series(v, name="baseline_adjustment", dtype="float64")
                for k, v in params.items()
            }
        )
        factor.index.names = ["group", "tretspef_grouped"]
        return self._update(factor)

    def waiting_list_adjustment(self):
        """Perform the waiting list adjustment.

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
        factor.index = pd.MultiIndex.from_tuples(
            [(True, i) for i in factor.index], names=["is_wla", "tretspef_grouped"]
        )
        factor.name = "waiting_list_adjustment"

        return self._update(factor)

    def non_demographic_adjustment(self):
        """Perform the non-demographic adjustment."""
        if not (params := self.run_params["non-demographic_adjustment"][self._activity_type]):
            return self

        match self.params["non-demographic_adjustment"]["value-type"]:
            case "year-on-year-growth":
                year_exponent = self.run_params["year"] - self.params["start_year"]
            case x:
                raise ValueError(f"invalid value-type: {x}")

        factor = pd.Series(params).rename("non-demographic_adjustment") ** year_exponent
        factor.index.names = ["ndggrp"]
        return self._update(factor)

    def apply_resampling(self):
        """Apply the row resampling to the data."""
        # get the random sampling for each row
        rng = self._model_iteration.rng
        factors = pd.concat(self.factors, axis=1)

        # reshape this to be the same as baseline counts
        overall_factor = (
            self._model_iteration.model.baseline_counts * factors.prod(axis=1).to_numpy()
        )

        row_samples = rng.poisson(overall_factor)

        step_counts = self._model_iteration.fix_step_counts(
            self.data, row_samples, factors, "model_interaction_term"
        ).assign(strategy="-")

        # apply the random sampling, update the data and get the counts
        data = self._model_iteration.model.apply_resampling(row_samples, self.data)

        return data, step_counts
