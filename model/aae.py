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
    def _expat_adjustment(data: pd.DataFrame, run_params: dict) -> pd.Series:
        expat_params = run_params["expat"]["aae"]
        join_cols = ["is_ambulance"]
        return data.merge(
            pd.DataFrame(
                [
                    {"is_ambulance": k == "ambulance", "value": v}
                    for k, v in expat_params.items()
                ]
            ).set_index(join_cols),
            how="left",
            left_on=join_cols,
            right_index=True,
        )["value"].fillna(1)

    @staticmethod
    def _repat_adjustment(data: pd.DataFrame, run_params: dict) -> pd.Series:
        repat_local_params = run_params["repat_local"]["aae"]
        repat_nonlocal_params = run_params["repat_nonlocal"]["aae"]
        join_cols = ["is_ambulance", "is_main_icb"]
        return data.merge(
            pd.DataFrame(
                [
                    {"is_ambulance": k1 == "ambulance", "is_main_icb": icb, "value": v1}
                    for (k0, icb) in [
                        (repat_local_params, True),
                        (repat_nonlocal_params, False),
                    ]
                    for k1, v1 in k0.items()
                ]
            ).set_index(join_cols),
            how="left",
            left_on=join_cols,
            right_index=True,
        )["value"].fillna(1)

    @staticmethod
    def _baseline_adjustment(data: pd.DataFrame, run_params: dict) -> pd.Series:
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

        # extract the parameters
        params = run_params["baseline_adjustment"]["aae"]

        # get the parameter value for each row
        return np.array(
            [params["ambulance" if k else "walk-in"] for k in data["is_ambulance"]]
        )

    @staticmethod
    def _step_counts(arrivals: pd.Series, arrivals_after: float, factors: dict) -> dict:
        cf = {}
        # as each row has a different weight attached to it, we need to use the factors
        # applied to each row individually
        n = arrivals
        arrivals_before = ns = n.sum()
        for k, v in factors.items():
            n2 = n * v
            n2s = n2.sum()
            cf[k] = n2s - ns
            n, ns = n2, n2s

        slack = 1 + (arrivals_after - ns) / (ns - arrivals_before)
        return {"baseline": arrivals_before, **{k: v * slack for k, v in cf.items()}}

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

        # get the factors
        factors = {
            "health_status_adjustment": hsa_f,
            "population_factors": demo_f[data["rn"]].to_numpy(),
            "expatriation": self._expat_adjustment(data, run_params).to_numpy(),
            "repatriation": self._repat_adjustment(data, run_params).to_numpy(),
            "baseline_adjustment": self._baseline_adjustment(data, run_params),
            "low_cost_dischaged": self._low_cost_discharged(data, params),
            "left_before_seen": self._left_before_seen(data, params),
            "frequent_attenders": self._frequent_attenders(data, params),
        }
        reduced_factor = np.prod(list(factors.values()), axis=0)
        arrivals_after = rng.poisson(data["arrivals"] * reduced_factor)

        # return the data
        change_factors = (
            pd.Series(
                self._step_counts(data["arrivals"], arrivals_after.sum(), factors)
            )
            .to_frame("value")
            .reset_index()
            .rename(columns={"index": "change_factor"})
            .assign(strategy="-", measure="arrivals")[
                ["change_factor", "strategy", "measure", "value"]
            ]
        )
        data["arrivals"] = arrivals_after
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
        model_results.loc[model_results["is_ambulance"], "measure"] = "ambulance"
        model_results.rename(columns={"arrivals": "value"}, inplace=True)

        # summarise the results to make the create_agg steps quicker
        model_results = model_results.groupby(
            # note: any columns used in the calls to _create_agg, including pod and measure
            # must be included below
            ["pod", "sitetret", "measure", "sex", "age_group"],
            as_index=False,
        ).agg({"value": "sum"})

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
