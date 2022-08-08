"""
Outpatients Module

Implements the Outpatients model.
"""

import os
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from model.model import Model


class OutpatientsModel(Model):
    """
    Outpatients Model

    Implementation of the Model for Outpatient attendances.

    :param params: the parameters to run the model with
    :param data_path: the path to where the data files live
    """

    def __init__(self, params: list, data_path: str) -> None:
        # call the parent init function
        super().__init__("op", params, data_path)

    def _waiting_list_adjustment(self, data: pd.DataFrame) -> pd.Series:
        """Create a series of factors for waiting list adjustment.

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame

        :returns: an array of factors, the length of data
        :rtype: numpy.ndarray
        """
        tretspef_n = data.groupby("tretspef").size()
        wla_param = pd.Series(self.params["waiting_list_adjustment"]["op"])
        wla = (wla_param / tretspef_n).fillna(0) + 1
        return wla[data.tretspef].to_numpy()

    def _followup_reduction(self, data: pd.DataFrame, run_params: dict) -> np.ndarray:
        """Followup Appointment Reduction

        Returns the factor values for the Followup Appointment Reduction strategy

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: an array of factors, the length of data
        :rtype: numpy.ndarray
        """
        return self._factor_helper(
            data, run_params["followup_reduction"], {"has_procedures": 0, "is_first": 0}
        )

    def _consultant_to_consultant_reduction(
        self, data: pd.DataFrame, run_params: dict
    ) -> np.ndarray:
        """Consultant to Consultant Referral Reduction


        Returns the factor values for the Consultant to Consultant Reduction strategy

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: an array of factors, the length of data
        :rtype: numpy.ndarray
        """
        return self._factor_helper(
            data,
            run_params["consultant_to_consultant_reduction"],
            {"is_cons_cons_ref": 1},
        )

    @staticmethod
    def _convert_to_tele(
        rng: np.random.Generator, data: pd.DataFrame, run_params: dict
    ) -> None:
        """Convert attendances to tele-attendances

        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict
        """
        # temp disable chained assignment warnings
        options = pd.get_option("mode.chained_assignment")
        pd.set_option("mode.chained_assignment", None)
        # get the parameters
        params = run_params["convert_to_tele"]
        # find locations of rows that didn't have procedures
        npix = ~data["has_procedures"]
        # create a value for converting attendances into tele attendances for each row
        # the value will be a random binomial value, i.e. we will convert between 0 and attendances
        # into tele attendances
        tele_conversion = rng.binomial(
            data.loc[npix, "attendances"], [params[t] for t in data.loc[npix, "type"]]
        )
        # update the columns, subtracting tc from one, adding tc to the other (we maintain the
        # number of overall attendances)
        data.loc[npix, "attendances"] -= tele_conversion
        data.loc[npix, "tele_attendances"] += tele_conversion
        # restore chained assignment warnings
        pd.set_option("mode.chained_assignment", options)

    def _run_poisson_step(
        self,
        rng: np.random.BitGenerator,
        data: pd.DataFrame,
        name: str,
        factor: np.ndarray,
        step_counts: dict,
    ) -> None:
        """Run a poisson step

        Resample the rows of `data` using a randomly generated poisson value for each row from
        `factor`.

        Updates the `step_counts` dictionary as a side effect.

        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param name: the name of the step (inserted into step_counts)
        :type name: str
        :param factor: a series with as many values as rows in `data` which will be used as the lambda
            value for the poisson distribution
        :type factor: pandas.Series
        :param step_counts: a dictionary containing the changes to measures for this step
        :type step_counts: dict

        :returns: the updated DataFrame
        :rtype: pandas.DataFrame
        """
        data.loc[:, "attendances"] = rng.poisson(
            data["attendances"].to_numpy() * factor
        )
        data.loc[:, "tele_attendances"] = rng.poisson(
            data["tele_attendances"].to_numpy() * factor
        )
        # remove rows where the overall number of attendances was 0
        data.drop(
            data[data["attendances"] + data["tele_attendances"] == 0].index,
            inplace=True,
        )
        # update the step count values
        self._update_step_counts(data, name, step_counts)

    def _run_binomial_step(
        self,
        rng: np.random.BitGenerator,
        data: pd.DataFrame,
        name: str,
        factor: np.ndarray,
        step_counts: dict,
    ) -> None:
        """Run a binomial step

        Resample the rows of `data` using a randomly generated binomial value for each row from
        `factor`.

        Updates the `step_counts` dictionary as a side effect.

        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param name: the name of the step (inserted into step_counts)
        :type name: str
        :param factor: a series with as many values as rows in `data` which will be used as the
            parameter of the distribution
        :type factor: pandas.Series
        :param step_counts: a dictionary containing the changes to measures for this step
        :type step_counts: dict

        :returns: the updated DataFrame
        :rtype: pandas.DataFrame
        """
        data.loc[:, "attendances"] = rng.binomial(
            data["attendances"].to_numpy(), factor
        )
        data.loc[:, "tele_attendances"] = rng.binomial(
            data["tele_attendances"].to_numpy(), factor
        )
        # remove rows where the overall number of attendances was 0
        data.drop(
            data[data["attendances"] + data["tele_attendances"] == 0].index,
            inplace=True,
        )
        # update the step count values
        self._update_step_counts(data, name, step_counts)

    @staticmethod
    def _update_step_counts(data: pd.DataFrame, name: str, step_counts: dict) -> None:
        """Update the dictionary of step counts

        Helper method for updating the step counts dictionary, which keeps track of the changes to
        the data after performing each step

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param name: the name of the step we just performed
        :type name: str
        :param step_counts: the dictionary that contains the step counts
        :type step_counts: dict
        """
        step_counts[name] = {
            k: sum(data[k]) - sum(v[k] for v in step_counts.values())
            for k in ["attendances", "tele_attendances"]
        }

    def _run(
        self,
        rng: np.random.Generator,
        data: pd.DataFrame,
        run_params: dict,
        demo_f: pd.Series,
        hsa_f: pd.Series,
    ) -> tuple[dict, pd.DataFrame]:
        """Run the model

        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict
        :param demo_f: the demographic adjustment factors
        :type demo_f: pandas.Series
        :param hsa_f: the health status adjustment factors
        :type hsa_f: pandas.Series

        :returns: a tuple containing the change factors DataFrame and the mode results DataFrame
        :rtype: (dict, pandas.DataFrame)
        """
        params = run_params["outpatient_factors"]
        step_counts = {
            "baseline": {k: data[k].sum() for k in ["attendances", "tele_attendances"]}
        }

        # captue current pandas options, and set chainged assignment off
        pd_options = pd.set_option("mode.chained_assignment", None)
        # first, run hsa as we have the factor already created
        self._run_poisson_step(
            rng, data, "health_status_adjustment", hsa_f, step_counts
        )
        # then, demographic modelling
        self._run_poisson_step(
            rng, data, "population_factors", demo_f[data["rn"]], step_counts
        )
        # waiting list adjustments
        self._run_poisson_step(
            rng,
            data,
            "waiting_list_adjustment",
            self._waiting_list_adjustment(data),
            step_counts,
        )
        # now run strategies
        self._run_binomial_step(
            rng,
            data,
            "followup_reduction",
            self._followup_reduction(data, params),
            step_counts,
        )
        self._run_binomial_step(
            rng,
            data,
            "consultant_to_consultant_referrals",
            self._consultant_to_consultant_reduction(data, params),
            step_counts,
        )
        # now run_step's are finished, restore pandas options
        pd.set_option("mode.chained_assignment", pd_options)
        # convert attendances to tele attendances
        self._convert_to_tele(rng, data, params)
        self._update_step_counts(data, "tele_conversion", step_counts)
        # return the data
        change_factors = (
            pd.DataFrame.from_dict(step_counts, orient="index")
            .rename_axis("change_factor")
            .reset_index()
            .assign(strategy="-")
            .melt(
                ["change_factor", "strategy"],
                ["attendances", "tele_attendances"],
                "measure",
            )
        )
        change_factors["value"] = change_factors["value"].astype(int)
        return (change_factors, data.drop(["hsagrp"], axis="columns"))

    def aggregate(self, model_results: pd.DataFrame, model_run: int) -> dict:
        """Aggregate the model results

        Can also be used to aggregate the baseline data by passing in the raw data

        :param model_results: a DataFrame containing the results of a model iteration
        :type model_results: pandas.DataFrame
        :param model_run: the current model run
        :type model_run: int

        :returns: a dictionary containing the different aggregations of this data
        :rtype: dict
        """
        model_results.loc[model_results["is_first"], "pod"] = "op_first"
        model_results.loc[~model_results["is_first"], "pod"] = "op_follow-up"
        model_results.loc[model_results["has_procedures"], "pod"] = "op_procedure"

        measures = model_results.melt(
            ["rn"], ["attendances", "tele_attendances"], "measure"
        )
        model_results = model_results.drop(
            ["attendances", "tele_attendances"], axis="columns"
        ).merge(measures, on="rn")

        agg = partial(self._create_agg, model_results)
        return {
            **agg(),
            **agg(["sex", "age_group"]),
            **agg(["sex", "tretspef"]),
        }

    def save_results(self, model_results: pd.DataFrame, path_fn: Callable[[str], str]) -> None:
        """Save the results of running the model

        This method is used for saving the results of the model run to disk as a parquet file.
        It saves just the `rn` (row number) column and the `arrivals`, with the intention that
        you rejoin to the original data.

        :param model_results: a DataFrame containing the results of a model iteration
        :param path_fn: a function which takes the activity type and returns a path
        """
        results.set_index(["rn"])[["attendances", "tele_attendances"]].to_parquet(
            f"{path_fn('op')}/0.parquet"
        )
