"""
Accident and Emergency Module

Implements the A&E model.
"""

from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.data import Data
from model.model import Model
from model.model_iteration import ModelIteration


class AaEModel(Model):
    """Accident and Emergency Model

    Implementation of the Model for Accident and Emergency attendances.

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
        data: Data,
        hsa: Any = None,
        run_params: dict = None,
        save_full_model_results: bool = False,
    ) -> None:
        # call the parent init function
        super().__init__(
            "aae",
            ["arrivals"],
            params,
            data,
            hsa,
            run_params,
            save_full_model_results,
        )

    def _get_data(self) -> pd.DataFrame:
        return self._data_loader.get_aae()

    def _add_pod_to_data(self) -> None:
        """Adds the POD column to data"""
        self.data["pod"] = "aae_type-" + self.data["aedepttype"]

    def get_data_counts(self, data: pd.DataFrame) -> npt.ArrayLike:
        """Get row counts of data

        :param data: the data to get the counts of
        :type data: pd.DataFrame
        :return: the counts of the data, required for activity avoidance steps
        :rtype: npt.ArrayLike
        """
        return np.array([data["arrivals"]]).astype(float)

    def _load_strategies(self) -> None:
        """Loads the activity mitigation strategies"""
        data = self.data.set_index("rn")
        self.strategies = {
            "activity_avoidance": pd.concat(
                [
                    data[data[c]]["hsagrp"].str.replace("aae", n).rename("strategy")
                    for (c, n) in [
                        ("is_frequent_attender", "frequent_attenders"),
                        ("is_left_before_treatment", "left_before_seen"),
                        ("is_low_cost_referred_or_discharged", "low_cost_discharged"),
                        ("is_discharged_no_treatment", "discharged_no_treatment"),
                    ]
                ]
            )
            .to_frame()
            .assign(sample_rate=1)
        }

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
        data["arrivals"] = row_samples[0]
        # return the altered data
        return data

    def efficiencies(self, data: pd.DataFrame, model_iteration: ModelIteration) -> None:
        """Run the efficiencies steps of the model

        :param model_iteration: an instance of the ModelIteration class
        :type model_iteration: model.model_iteration.ModelIteration
        """

        # A&E doesn't have any efficiencies steps
        return data, None

    @staticmethod
    def process_results(data: pd.DataFrame) -> pd.DataFrame:
        """Processes the data into a format suitable for aggregation in results files

        :param data: Data to be processed. Format should be similar to Model.data
        :type data: pd.DataFrame
        """
        data["measure"] = "walk-in"
        data.loc[data["is_ambulance"], "measure"] = "ambulance"
        data.rename(columns={"arrivals": "value"}, inplace=True)

        # summarise the results to make the create_agg steps quicker
        data = data.groupby(
            # note: any columns used in the calls to _create_agg, including pod and measure
            # must be included below
            [
                "pod",
                "sitetret",
                "acuity",
                "measure",
                "sex",
                "age",
                "age_group",
                "attendance_category",
            ],
            as_index=False,
        ).agg({"value": "sum"})
        return data

    def aggregate(self, model_iteration: ModelIteration) -> Tuple[Callable, dict]:
        """Aggregate the model results

        Can also be used to aggregate the baseline data by passing in a `ModelIteration` with
        the `model_run` argument set `-1`.

        :param model_iteration: an instance of the `ModelIteration` class
        :type model_iteration: model.model_iteration.ModelIteration

        :returns: a dictionary containing the different aggregations of this data
        :rtype: dict
        """
        model_results = self.process_results(model_iteration.get_model_results())

        return (
            model_results,
            [
                ["acuity"],
                ["attendance_category"],
            ],
        )

    def calculate_avoided_activity(
        self, data: pd.DataFrame, data_resampled: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate the rows that have been avoided

        :param data: The data before the binomial thinning step
        :type data: pd.DataFrame
        :return: The data that was avoided in the binomial thinning step
        :rtype: pd.DataFrame
        """
        avoided = data["arrivals"] - data_resampled["arrivals"]
        data["arrivals"] = avoided
        return data

    def save_results(
        self, model_iteration: ModelIteration, path_fn: Callable[[str], str]
    ) -> None:
        """Save the results of running the model

        This method is used for saving the results of the model run to disk as a parquet file.
        It saves just the `rn` (row number) column and the `arrivals`, with the intention that
        you rejoin to the original data.

        :param model_iteration: an instance of the `ModelIteration` class
        :type model_iteration: model.model_iteration.ModelIteration
        :param path_fn: a function which takes the activity type and returns a path
        :type path_fn: Callable[[str], str]
        """
        model_iteration.get_model_results().set_index(["rn"])[["arrivals"]].to_parquet(
            f"{path_fn('aae')}/0.parquet"
        )
        model_iteration.avoided_activity.set_index(["rn"])[["arrivals"]].to_parquet(
            f"{path_fn('aae_avoided')}/0.parquet"
        )
