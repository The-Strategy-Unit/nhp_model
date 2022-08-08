"""
Accident and Emergency Module

Implements the A&E model.
"""

import os
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from model.model import Model


class AaEModel(Model):
    """Accident and Emergency Model

    Implementation of the Model for Accident and Emergency attendances.

    :param params: the parameters to run the model with
    :param data_path: the path to where the data files live
    """

    def __init__(self, params: list, data_path: str) -> None:
        # call the parent init function
        super().__init__("aae", params, data_path)

    def _low_cost_discharged(self, data: pd.DataFrame, run_params: dict) -> np.ndarray:
        """Low Cost Discharge Reduction

        Returns the factor values for the Low Cost Discharge Reduction strategy

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: an array of factors, the length of data
        :rtype: numpy.ndarray
        """
        return self._factor_helper(
            data,
            run_params["low_cost_discharged"],
            {"is_low_cost_referred_or_discharged": 1},
        )

    def _left_before_seen(self, data: pd.DataFrame, run_params: dict) -> np.ndarray:
        """Left Before Seen Reduction

        Returns the factor values for the Left Before Seen Reduction strategy

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: an array of factors, the length of data
        :rtype: numpy.ndarray
        """
        return self._factor_helper(
            data, run_params["left_before_seen"], {"is_left_before_treatment": 1}
        )

    def _frequent_attenders(self, data: pd.DataFrame, run_params: dict) -> np.ndarray:
        """Frequent Attenders Reduction

        Returns the factor values for the Frequent Attenders Reduction strategy

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: an array of factors, the length of data
        :rtype: numpy.ndarray
        """
        return self._factor_helper(
            data, run_params["frequent_attenders"], {"is_frequent_attender": 1}
        )

    @staticmethod
    def _run_poisson_step(
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
        # perform the step
        data["arrivals"] = rng.poisson(data["arrivals"].to_numpy() * factor)
        # remove rows where the overall number of attendances was 0
        data.drop(data[data["arrivals"] == 0].index, inplace=True)
        # update the step counts
        step_counts[name] = sum(data["arrivals"]) - sum(step_counts.values())

    @staticmethod
    def _run_binomial_step(
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
        # perform the step
        data["arrivals"] = rng.binomial(data["arrivals"].to_numpy(), factor)
        # remove rows where the overall number of attendances was 0
        data.drop(data[data["arrivals"] == 0].index, inplace=True)
        # update the step counts
        step_counts[name] = sum(data["arrivals"]) - sum(step_counts.values())

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
        params = run_params["aae_factors"]

        step_counts = {"baseline": sum(data["arrivals"])}

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
        # now run strategies
        self._run_binomial_step(
            rng,
            data,
            "low_cost_dischaged",
            self._low_cost_discharged(data, params),
            step_counts,
        )
        self._run_binomial_step(
            rng,
            data,
            "left_before_seen",
            self._left_before_seen(data, params),
            step_counts,
        )
        self._run_binomial_step(
            rng,
            data,
            "frequent_attenders",
            self._frequent_attenders(data, params),
            step_counts,
        )
        # now run_step's are finished, restore pandas options
        pd.set_option("mode.chained_assignment", pd_options)
        # return the data
        change_factors = (
            pd.Series(step_counts)
            .to_frame("value")
            .reset_index()
            .rename(columns={"index": "change_factor"})
            .assign(strategy="-", measure="arrivals")[
                ["change_factor", "strategy", "measure", "value"]
            ]
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
        model_results["pod"] = "aae_type-" + model_results["aedepttype"]
        model_results["measure"] = "walk-in"
        model_results.loc[
            model_results["aearrivalmode"] == "1", "measure"
        ] = "ambulance"
        model_results.rename(columns={"arrivals": "value"}, inplace=True)

        agg = partial(self._create_agg, model_results)
        return {**agg(), **agg(["sex", "age_group"])}

    def save_results(
        self, model_results: pd.DataFrame, path_fn: Callable[[str], str]
    ) -> None:
        """Save the results of running the model

        This method is used for saving the results of the model run to disk as a parquet file.
        It saves just the `rn` (row number) column and the `arrivals`, with the intention that
        you rejoin to the original data.

        :param model_results: a DataFrame containing the results of a model iteration
        :param path_fn: a function which takes the activity type and returns a path
        """
        model_results.set_index(["rn"])[["arrivals"]].to_parquet(
            f"{path_fn('aae')}/0.parquet"
        )
