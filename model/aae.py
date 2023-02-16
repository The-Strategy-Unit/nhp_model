"""
Accident and Emergency Module

Implements the A&E model.
"""

from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from model.model import Model
from model.model_run import ModelRun
from model.row_resampling import RowResampling


class AaEModel(Model):
    """Accident and Emergency Model

    Implementation of the Model for Accident and Emergency attendances.

    :param params: the parameters to run the model with
    :param data_path: the path to where the data files live
    """

    def __init__(self, params: list, data_path: str) -> None:
        # call the parent init function
        super().__init__("aae", params, data_path)
        self.data["group"] = np.where(self.data["is_ambulance"], "ambulance", "walk-in")
        self.data["tretspef"] = "Other"

        self._baseline_counts = np.array([self.data["arrivals"]]).astype(float)

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

    def _run(self, model_run: ModelRun) -> tuple[dict, pd.DataFrame]:
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
        data, step_counts = (
            RowResampling(model_run, self._baseline_counts)
            .demographic_adjustment()
            .health_status_adjustment()
            .expat_adjustment()
            .repat_adjustment()
            .baseline_adjustment()
            .apply_resampling()
        )
        # return the data
        change_factors = (
            pd.Series({k: v[0] for k, v in step_counts.items()})
            .to_frame("value")
            .rename_axis(["change_factor", "strategy"])
            .reset_index()
            .assign(measure="arrivals")
        )
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
