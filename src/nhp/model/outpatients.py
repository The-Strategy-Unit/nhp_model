"""Outpatients Module

Implements the Outpatients model.
"""

from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from nhp.model.data import Data
from nhp.model.model import Model
from nhp.model.model_iteration import ModelIteration


class OutpatientsModel(Model):
    """Outpatients Model

    Implementation of the Model for Outpatient attendances.

    :param params: the parameters to run the model with, or the path to a params file to load
    :type params: dict or string
    :param data: a Data class ready to be constructed
    :type data: Data
    :param hsa: An instance of the HealthStatusAdjustment class. If left as None an instance is
    created
    :type hsa: HealthStatusAdjustment, optional
    :param run_params: the parameters to use for each model run. generated automatically if left as
    None
    :type run_params: dict
    :param save_full_model_results: whether to save the full model results or not
    :type save_full_model_results: bool, optional
    """

    def __init__(
        self,
        params: dict,
        data: Callable[[int, str], Data],
        hsa: Any = None,
        run_params: dict | None = None,
        save_full_model_results: bool = False,
    ) -> None:
        # call the parent init function
        super().__init__(
            "op",
            ["attendances", "tele_attendances"],
            params,
            data,
            hsa,
            run_params,
            save_full_model_results,
        )

    def _get_data(self, data_loader: Data) -> pd.DataFrame:
        return data_loader.get_op()

    def _add_pod_to_data(self) -> None:
        """Adds the POD column to data"""
        self.data.loc[self.data["is_first"], "pod"] = "op_first"
        self.data.loc[~self.data["is_first"], "pod"] = "op_follow-up"
        self.data.loc[self.data["has_procedures"], "pod"] = "op_procedure"

    def get_data_counts(self, data) -> npt.ArrayLike:
        return (
            data[["attendances", "tele_attendances"]]
            .to_numpy()
            .astype(float)
            .transpose()
        )

    def _load_strategies(self, data_loader: Data) -> None:
        data = self.data.set_index("rn")

        self.strategies = {
            "activity_avoidance": pd.concat(
                [
                    "followup_reduction_"
                    + data[~data["is_first"] & ~data["has_procedures"]]["type"],
                    "consultant_to_consultant_reduction_"
                    + data[data["is_cons_cons_ref"]]["type"],
                    "gp_referred_first_attendance_reduction_"
                    + data[data["is_gp_ref"] & data["is_first"]]["type"],
                ]
            )
            .rename("strategy")
            .to_frame()
            .assign(sample_rate=1),
            "efficiencies": pd.concat(
                ["convert_to_tele_" + data[~data["has_procedures"]]["type"]]
            ),
        }

    @staticmethod
    def _convert_to_tele(
        data,
        model_iteration: ModelIteration,
    ) -> None:
        """Convert attendances to tele-attendances

        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict
        """
        # TODO: we need to make sure efficiences contains convert to tele keys
        rng = model_iteration.rng
        params = model_iteration.run_params["efficiencies"]["op"]
        strategies = model_iteration.model.strategies["efficiencies"]
        # make sure to take the complement of the parameter
        factor = 1 - data["rn"].map(strategies.map(params)).fillna(1)
        # create a value for converting attendances into tele attendances for each row
        # the value will be a random binomial value, i.e. we will convert between 0 and attendances
        # into tele attendances
        tele_conversion = rng.binomial(data["attendances"].to_list(), factor.to_list())
        # update the columns, subtracting tc from one, adding tc to the other (we maintain the
        # number of overall attendances)
        data["attendances"] -= tele_conversion
        data["tele_attendances"] += tele_conversion

        step_counts = (
            pd.DataFrame(
                {
                    "pod": data["pod"],
                    "sitetret": data["sitetret"],
                    "change_factor": "efficiencies",
                    "strategy": "convert_to_tele",
                    "attendances": tele_conversion * -1,
                    "tele_attendances": tele_conversion,
                }
            )
            .groupby(["pod", "sitetret", "change_factor", "strategy"], as_index=False)
            .sum()
            .query("attendances<0")
        )
        return data, step_counts

    def apply_resampling(
        self, row_samples: npt.ArrayLike, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply row resampling

        Called from within `model.activity_resampling.ActivityResampling.apply_resampling`

        :param row_samples: [1xn] array, where n is the number of rows in `data`, containing the new
        values for `data["arrivals"]`
        :type row_samples: npt.ArrayLike
        :param data: the data that we want to update
        :type data: pd.DataFrame
        :return: the updated data
        :rtype: pd.DataFrame
        """
        data["attendances"] = row_samples[0]
        data["tele_attendances"] = row_samples[1]
        # return the altered data
        return data

    def efficiencies(self, data: pd.DataFrame, model_iteration: ModelIteration) -> None:
        """Run the efficiencies steps of the model

        :param model_iteration: an instance of the ModelIteration class
        :type model_iteration: model.model_iteration.ModelIteration
        """
        data, step_counts = self._convert_to_tele(data, model_iteration)
        return data, step_counts

    def calculate_avoided_activity(
        self, data: pd.DataFrame, data_resampled: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate the rows that have been avoided

        :param data: The data before the binomial thinning step
        :type data: pd.DataFrame
        :return: The data that was avoided in the binomial thinning step
        :rtype: pd.DataFrame
        """
        avoided = (
            data[["attendances", "tele_attendances"]]
            - data_resampled[["attendances", "tele_attendances"]]
        )
        data[["attendances", "tele_attendances"]] = avoided
        return data

    @staticmethod
    def process_results(data: pd.DataFrame) -> pd.DataFrame:
        """Processes the data into a format suitable for aggregation in results files

        :param data: Data to be processed. Format should be similar to Model.data
        :type data: pd.DataFrame
        """
        measures = data.melt(["rn"], ["attendances", "tele_attendances"], "measure")
        data = (
            data.drop(["attendances", "tele_attendances"], axis="columns")
            .merge(measures, on="rn")
            # summarise the results to make the create_agg steps quicker
            .groupby(
                # note: any columns used in the calls to _create_agg, including pod and measure
                # must be included below
                [
                    "pod",
                    "sitetret",
                    "measure",
                    "sex",
                    "age_group",
                    "age",
                    "tretspef",
                    "tretspef_grouped",
                ],
                dropna=False,
                as_index=False,
            )
            .agg({"value": "sum"})
            .fillna("unknown")
        )
        return data

    def aggregate(self, model_iteration: ModelIteration) -> Tuple[Callable, dict]:
        """Aggregate the model results

        Can also be used to aggregate the baseline data by passing in the raw data

        :param model_results: a DataFrame containing the results of a model iteration
        :type model_results: pandas.DataFrame
        :param model_run: the current model run
        :type model_run: int

        :returns: a dictionary containing the different aggregations of this data
        :rtype: dict
        """
        model_results = self.process_results(model_iteration.get_model_results())

        return (
            model_results,
            [
                ["sex", "tretspef_grouped"],
                ["tretspef"],
            ],
        )

    def save_results(
        self, model_iteration: ModelIteration, path_fn: Callable[[str], str]
    ) -> None:
        """Save the results of running the model

        This method is used for saving the results of the model run to disk as a parquet file.
        It saves just the `rn` (row number) column and the `attendances` and `tele_attendances
        columns, with the intention that you rejoin to the original data.

        :param model_results: a DataFrame containing the results of a model iteration
        :param path_fn: a function which takes the activity type and returns a path
        """
        model_iteration.get_model_results().set_index(["rn"])[
            ["attendances", "tele_attendances"]
        ].to_parquet(f"{path_fn('op')}/0.parquet")

        model_iteration.avoided_activity.set_index(["rn"])[
            ["attendances", "tele_attendances"]
        ].to_parquet(f"{path_fn('op_avoided')}/0.parquet")
