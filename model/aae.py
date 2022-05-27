"""
Accident and Emergency Module

Implements the A&E model.
"""

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

        sc_a = sum(data["arrivals"])
        step_counts = {
            "baseline": pd.DataFrame({"measure": ["arrivals"], "value": [sc_a]}, ["-"])
        }

        def update_step_counts(name):
            nonlocal data, step_counts, sc_a
            sc_ap = sum(data["arrivals"])
            step_counts[name] = pd.DataFrame(
                {"measure": ["arrivals"], "value": [sc_ap - sc_a]}, ["-"]
            )
            # replace the values
            sc_a = sc_ap

        def run_poisson_step(factor, name):
            nonlocal data
            # perform the step
            data["arrivals"] = rng.poisson(data["arrivals"].to_numpy() * factor)
            # remove rows where the overall number of attendances was 0
            data = data[data["arrivals"] > 0]
            # update the step count values
            update_step_counts(name)

        def run_binomial_step(factor, name):
            nonlocal data
            # perform the step
            data["arrivals"] = rng.binomial(data["arrivals"].to_numpy(), factor)
            # remove rows where the overall number of attendances was 0
            data = data[data["arrivals"] > 0]
            # update the step count values
            update_step_counts(name)

        # captue current pandas options, and set chainged assignment off
        pd_options = pd.set_option("mode.chained_assignment", None)
        # first, run hsa as we have the factor already created
        run_poisson_step(hsa_f, "health_status_adjustment")
        # then, demographic modelling
        run_poisson_step(aav_f[data["rn"]], "population_factors")
        # now run strategies
        run_binomial_step(
            self._low_cost_discharged(data, params), "low_cost_discharged"
        )
        run_binomial_step(self._left_before_seen(data, params), "left_before_seen")
        run_binomial_step(self._frequent_attenders(data, params), "frequent_attenders")
        # now run_step's are finished, restore pandas options
        pd.set_option("mode.chained_assignment", pd_options)
        # return the data
        change_factors = (
            pd.concat(step_counts)
            .rename_axis(["change_factor", "strategy"])
            .reset_index()
        )
        change_factors["value"] = change_factors["value"].astype(int)
        return (change_factors, data.drop(["hsagrp"], axis="columns"))

    def aggregate(self, model_results: pd.DataFrame) -> dict:
        """
        Aggregate the model results

        * model_results: a DataFrame containing the results of a model iteration

        returns: a dictionary containing the different aggregations of this data

        Can also be used to aggregate the baseline data by passing in the raw data
        """
        model_results["pod"] = "aae_type-" + model_results["aedepttype"]
        model_results["measure"] = "walk-in"
        model_results.loc[
            model_results["aearrivalmode"] == "1", "measure"
        ] = "ambulance"
        model_results.rename(columns={"arrivals": "value"}, inplace=True)

        return {
            **self._create_agg(model_results),
            **self._create_agg(model_results, ["sex", "age_group"]),
        }
