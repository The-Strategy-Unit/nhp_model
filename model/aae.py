"""
Accident and Emergency Module

Implements the A&E model.
"""

from model.model import Model


class AaEModel(Model):
    """
    Accident and Emergency Model

    * results_path: where the data is stored

    Implements the model for accident and emergency data. See `Model()` for documentation on the
    generic class.
    """

    def __init__(self, results_path):
        # call the parent init function
        Model.__init__(self, "aae", results_path)

    #
    def _low_cost_discharged(self, data, run_params):
        return self._factor_helper(
            data,
            run_params["low_cost_discharged"],
            {"is_low_cost_referred_or_discharged": 1},
        )

    #
    def _left_before_seen(self, data, run_params):
        return self._factor_helper(
            data, run_params["left_before_seen"], {"is_left_before_treatment": 1}
        )

    #
    def _frequent_attenders(self, data, run_params):
        return self._factor_helper(
            data, run_params["frequent_attenders"], {"is_frequent_attender": 1}
        )

    #
    def _run(self, rng, data, run_params, hsa_f):
        """
        Run the model once

        returns: a tuple of the selected varient and the updated DataFrame
        """
        params = run_params["aae_factors"]
        # create a single factor for how many times to select that row
        factor = (
            data["factor"].to_numpy()
            * hsa_f
            * self._low_cost_discharged(data, params)
            * self._left_before_seen(data, params)
            * self._frequent_attenders(data, params)
        )
        data["arrivals"] = rng.poisson(data["arrivals"] * factor)
        data = data[data["arrivals"] > 0]
        # return the data
        return data[["arrivals"]].reset_index()
