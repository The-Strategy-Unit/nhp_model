"""
Accident and Emergency Module

Implements the A&E model.
"""

import numpy as np
import pandas as pd

from model.helpers import age_groups
from model.model import Model


class AaEModel(Model):
    """
    Accident and Emergency Model

    * results_path: where the data is stored

    Implements the model for accident and emergency data. See `Model()` for documentation on the
    generic class.
    """

    def __init__(self, params, data_path: str):
        # call the parent init function
        Model.__init__(self, "aae", params, data_path)

    def _low_cost_discharged(self, data, run_params):
        return self._factor_helper(
            data,
            run_params["low_cost_discharged"],
            {"is_low_cost_referred_or_discharged": 1},
        )

    def _left_before_seen(self, data, run_params):
        return self._factor_helper(
            data, run_params["left_before_seen"], {"is_left_before_treatment": 1}
        )

    def _frequent_attenders(self, data, run_params):
        return self._factor_helper(
            data, run_params["frequent_attenders"], {"is_frequent_attender": 1}
        )

    def _run(
        self, rng, data, run_params, aav_f, hsa_f
    ):  # pylint: disable=too-many-arguments
        """
        Run the model once

        returns: a tuple of the selected varient and the updated DataFrame
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
        # before we do anything, reset the index to keep the row number
        data.reset_index(inplace=True)
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

    @staticmethod
    def aggregate(model_results):
        """
        Aggregate the model results
        """
        model_results["age_group"] = age_groups(model_results["age"])

        model_results["pod"] = "aae_type-" + model_results["aedepttype"]
        model_results["measure"] = "walk-in"
        model_results.loc[
            model_results["aearrivalmode"] == "1", "measure"
        ] = "ambulance"
        model_results["tretspef"] = "Other"

        def agg(cols):
            return (
                model_results.groupby(cols + ["measure"], as_index=False)
                .agg({"arrivals": np.sum})
                .rename(columns={"arrivals": "value"})
                .to_dict("records")
            )

        return {
            "default": agg(["pod"]),
            "sex+age_group": agg(["pod", "sex", "age_group"]),
            "sex+tretspef": agg(["pod", "sex", "tretspef"]),
        }
