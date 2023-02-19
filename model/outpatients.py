"""
Outpatients Module

Implements the Outpatients model.
"""

from functools import partial
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.activity_avoidance import ActivityAvoidance
from model.model import Model
from model.model_run import ModelRun


class OutpatientsModel(Model):
    """
    Outpatients Model

    Implementation of the Model for Outpatient attendances.

    :param params: the parameters to run the model with
    :param data_path: the path to where the data files live
    """

    def __init__(self, params: list, data_path: str) -> None:
        # initialise values for testing purposes
        self.data = None
        # call the parent init function
        super().__init__("op", params, data_path)
        self._update_baseline_data()
        self._baseline_counts = self._get_data_counts(self.data)
        self._load_strategies()

    def _update_baseline_data(self) -> None:
        self.data["group"] = np.where(
            self.data["has_procedures"],
            "procedure",
            np.where(self.data["is_first"], "first", "followup"),
        )
        self.data["is_wla"] = True

    def _get_data_counts(self, data) -> npt.ArrayLike:
        return (
            self.data[["attendances", "tele_attendances"]]
            .to_numpy()
            .astype(float)
            .transpose()
        )

    def _load_strategies(self):
        data = self.data.set_index("rn")

        self._strategies = {
            "activity_avoidance": pd.concat(
                [
                    "followup_reduction_"
                    + data[~data["is_first"] & ~data["has_procedures"]]["type"],
                    "consultant_to_consultant_reduction_"
                    + data[data["is_cons_cons_ref"]]["type"],
                ]
            )
            .rename("strategy")
            .to_frame()
            .assign(sample_rate=1)
        }

    @staticmethod
    def _convert_to_tele(
        rng: np.random.Generator,
        data: pd.DataFrame,
        run_params: dict,
        step_counts: dict,
    ) -> None:
        """Convert attendances to tele-attendances

        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict
        """
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
        tele_conversion = tele_conversion.sum()
        step_counts[("efficiencies", "convert_to_tele")] = {
            "attendances": -tele_conversion,
            "tele_attendances": tele_conversion,
        }

    def _apply_resampling(self, row_samples, data):
        data["attendances"] = row_samples[0]
        data["tele_attendances"] = row_samples[1]
        # return the altered data and the amount of admissions/beddays after resampling
        return (data, self._get_data_counts(data))

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
            ActivityAvoidance(model_run, self._baseline_counts)
            .demographic_adjustment()
            .health_status_adjustment()
            .expat_adjustment()
            .repat_adjustment()
            .waiting_list_adjustment()
            .baseline_adjustment()
            .activity_avoidance()
            .apply_resampling()
        )

        step_counts = {
            k: {"attendances": v[0], "tele_attendances": v[1]}
            for k, v in step_counts.items()
            if k != "convert_to_tele"
        }

        # convert attendances to tele attendances
        self._convert_to_tele(
            model_run.rng, data, model_run.run_params["outpatient_factors"], step_counts
        )

        # return the data
        change_factors = (
            pd.DataFrame.from_dict(step_counts, orient="index")
            .rename_axis(["change_factor", "strategy"])
            .reset_index()
            .melt(
                ["change_factor", "strategy"],
                ["attendances", "tele_attendances"],
                "measure",
            )
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
        model_results.loc[model_results["is_first"], "pod"] = "op_first"
        model_results.loc[~model_results["is_first"], "pod"] = "op_follow-up"
        model_results.loc[model_results["has_procedures"], "pod"] = "op_procedure"

        measures = model_results.melt(
            ["rn"], ["attendances", "tele_attendances"], "measure"
        )
        model_results = (
            model_results.drop(["attendances", "tele_attendances"], axis="columns")
            .merge(measures, on="rn")
            # summarise the results to make the create_agg steps quicker
            .groupby(
                # note: any columns used in the calls to _create_agg, including pod and measure
                # must be included below
                ["pod", "sitetret", "measure", "sex", "age_group", "tretspef"],
                as_index=False,
            )
            .agg({"value": "sum"})
        )

        agg = partial(self._create_agg, model_results)
        return {
            **agg(),
            **agg(["sex", "age_group"]),
            **agg(["sex", "tretspef"]),
        }

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
        model_results.set_index(["rn"])[["attendances", "tele_attendances"]].to_parquet(
            f"{path_fn('op')}/0.parquet"
        )
