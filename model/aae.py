"""
Accident and Emergency Module

Implements the A&E model.
"""

import os
from functools import partial

import numpy as np
import pandas as pd

from model.model import Model


class AaEModel(Model):
    """
    Accident and Emergency Model

    * params: the parameters to run the model with
    * data_path: the path to where the data files live

    Inherits from the Model class.
    """

    def __init__(self, params: list, data_path: str) -> None:
        # call the parent init function
        super().__init__("aae", params, data_path)

    def _low_cost_discharged(self, data: pd.DataFrame, run_params: dict) -> np.ndarray:
        """Low Cost Discharge Reduction

        * data: the DataFrame that we are updating
        * run_params: the parameters to use for this model run (see `Model._get_run_params()`)

        returns: an array of factors, the length of data
        """
        return self._factor_helper(
            data,
            run_params["low_cost_discharged"],
            {"is_low_cost_referred_or_discharged": 1},
        )

    def _left_before_seen(self, data: pd.DataFrame, run_params: dict) -> np.ndarray:
        """Left Before Seen Reduction

        * data: the DataFrame that we are updating
        * run_params: the parameters to use for this model run (see `Model._get_run_params()`)

        returns: an array of factors, the length of data
        """
        return self._factor_helper(
            data, run_params["left_before_seen"], {"is_left_before_treatment": 1}
        )

    def _frequent_attenders(self, data: pd.DataFrame, run_params: dict) -> np.ndarray:
        """Frequen Attenders Reduction

        * data: the DataFrame that we are updating
        * run_params: the parameters to use for this model run (see `Model._get_run_params()`)

        returns: an array of factors, the length of data
        """
        return self._factor_helper(
            data, run_params["frequent_attenders"], {"is_frequent_attender": 1}
        )

    @staticmethod
    def _run_poisson_step(rng, data, name, factor, step_counts):
        # perform the step
        data["arrivals"] = rng.poisson(data["arrivals"].to_numpy() * factor)
        # remove rows where the overall number of attendances was 0
        data.drop(data[data["arrivals"] == 0].index, inplace=True)
        # update the step counts
        step_counts[name] = sum(data["arrivals"]) - sum(step_counts.values())

    @staticmethod
    def _run_binomial_step(rng, data, name, factor, step_counts):
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
        aav_f: pd.Series,
        hsa_f: pd.Series,
    ) -> tuple[pd.DataFrame, dict]:
        """Run the model

        * rng: an np.random.Generator, created for each model iteration
        * data: the DataFrame that we are updating
        * run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        * aav_f: the demographic adjustment factors
        * hsa_f: the health status adjustment factors

        returns: a tuple containing the change factors DataFrame and the mode results DataFrame
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
            rng, data, "population_factors", aav_f[data["rn"]], step_counts
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
        """
        Aggregate the model results

        * model_results: a DataFrame containing the results of a model iteration
        * model_run: the current model run

        returns: a dictionary containing the different aggregations of this data

        Can also be used to aggregate the baseline data by passing in the raw data
        """
        model_results["pod"] = "aae_type-" + model_results["aedepttype"]
        model_results["measure"] = "walk-in"
        model_results.loc[
            model_results["aearrivalmode"] == "1", "measure"
        ] = "ambulance"
        model_results.rename(columns={"arrivals": "value"}, inplace=True)

        agg = partial(self._create_agg, model_results)
        return {**agg(), **agg(["sex", "age_group"])}

    def save_results(self, results, path_fn):
        """Save the results of running the model"""
        results.set_index(["rn"])[["arrivals"]].to_parquet(f"{path_fn('aae')}/0.parquet")
