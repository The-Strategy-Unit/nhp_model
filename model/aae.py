"""
Accident and Emergency Module

Implements the A&E model.
"""

from functools import partial
from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.model import Model
from model.model_run import ModelRun


class AaEModel(Model):
    """Accident and Emergency Model

    Implementation of the Model for Accident and Emergency attendances.

    :param params: the parameters to run the model with
    :param data_path: the path to where the data files live
    """

    def __init__(
        self,
        params: dict,
        data_path: str,
        hsa: Any,
        run_params: dict = None,
        save_full_model_results: bool = False,
    ) -> None:
        # call the parent init function
        super().__init__(
            "aae",
            ["arrivals"],
            params,
            data_path,
            hsa,
            run_params,
            save_full_model_results,
        )

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

        Called from within `model.activity_avoidance.ActivityAvoidance.apply_resampling`

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

    def efficiencies(self, model_run: ModelRun) -> None:
        """Run the efficiencies steps of the model

        :param model_run: an instance of the ModelRun class
        :type model_run: model.model_run.ModelRun
        """

        # A&E doesn't have any efficiencies steps

    def aggregate(self, model_run: ModelRun) -> Tuple[Callable, dict]:
        """Aggregate the model results

        Can also be used to aggregate the baseline data by passing in a `ModelRun` with
        the `model_run` argument set `-1`.

        :param model_run: an instance of the `ModelRun` class
        :type model_run: model.model_run.ModelRun

        :returns: a dictionary containing the different aggregations of this data
        :rtype: dict
        """
        model_results = model_run.get_model_results()

        model_results["measure"] = "walk-in"
        model_results.loc[model_results["is_ambulance"], "measure"] = "ambulance"
        model_results.rename(columns={"arrivals": "value"}, inplace=True)

        # summarise the results to make the create_agg steps quicker
        model_results = model_results.groupby(
            # note: any columns used in the calls to _create_agg, including pod and measure
            # must be included below
            ["pod", "sitetret", "measure", "sex", "age", "age_group"],
            as_index=False,
        ).agg({"value": "sum"})

        agg = partial(self._create_agg, model_results)
        return (agg, {})

    def save_results(self, model_run: ModelRun, path_fn: Callable[[str], str]) -> None:
        """Save the results of running the model

        This method is used for saving the results of the model run to disk as a parquet file.
        It saves just the `rn` (row number) column and the `arrivals`, with the intention that
        you rejoin to the original data.

        :param model_run: an instance of the `ModelRun` class
        :type model_run: model.model_run.ModelRun
        :param path_fn: a function which takes the activity type and returns a path
        :type path_fn: Callable[[str], str]
        """
        model_run.get_model_results().set_index(["rn"])[["arrivals"]].to_parquet(
            f"{path_fn('aae')}/0.parquet"
        )
