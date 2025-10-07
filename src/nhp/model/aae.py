"""Accident and Emergency Module.

Implements the A&E model.
"""

from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd

from nhp.model.data import Data
from nhp.model.model import Model
from nhp.model.model_iteration import ModelIteration


class AaEModel(Model):
    """Accident and Emergency Model.

    Implementation of the Model for Accident and Emergency attendances.

    Args:
        params: The parameters to run the model with, or the path to a params file to load.
        data: A callable that creates a Data instance.
        hsa: An instance of the HealthStatusAdjustment class. If left as None an instance is
            created. Defaults to None.
        run_params: The parameters to use for each model run. Generated automatically if left as
            None. Defaults to None.
        save_full_model_results: Whether to save the full model results or not. Defaults to False.
    """

    def __init__(
        self,
        params: dict | str,
        data: Callable[[int, str], Data],
        hsa: Any = None,
        run_params: dict | None = None,
        save_full_model_results: bool = False,
    ) -> None:
        """Initialise the A&E Model.

        Args:
            params: The parameters to use.
            data: A method to create a Data instance.
            hsa: Health Status Adjustment object. Defaults to None.
            run_params: The run parameters to use. Defaults to None.
            save_full_model_results: Whether to save full model results. Defaults to False.
        """
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

    def _get_data(self, data_loader: Data) -> pd.DataFrame:
        return data_loader.get_aae()

    def get_data_counts(self, data: pd.DataFrame) -> np.ndarray:
        """Get row counts of data.

        Args:
            data: The data to get the counts of.

        Returns:
            The counts of the data, required for activity avoidance steps.
        """
        return np.array([data["arrivals"]]).astype(float)

    def _load_strategies(self, data_loader: Data) -> None:
        """Loads the activity mitigation strategies."""
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

    def apply_resampling(self, row_samples: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
        """Apply row resampling.

        Called from within `model.activity_resampling.ActivityResampling.apply_resampling`.

        Args:
            row_samples: [1xn] array, where n is the number of rows in `data`, containing the new
                values for `data["arrivals"]`.
            data: The data that we want to update.

        Returns:
            The updated data.
        """
        data["arrivals"] = row_samples[0]
        # return the altered data
        return data

    def efficiencies(
        self, data: pd.DataFrame, model_iteration: ModelIteration
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Run the efficiencies steps of the model.

        Args:
            data: The data to apply efficiencies to.
            model_iteration: An instance of the ModelIteration class.

        Returns:
            Tuple containing the updated data and step counts (None for A&E).
        """
        # A&E doesn't have any efficiencies steps
        return data, None

    @staticmethod
    def process_results(data: pd.DataFrame) -> pd.DataFrame:
        """Process the data into a format suitable for aggregation in results files.

        Args:
            data: Data to be processed. Format should be similar to Model.data.

        Returns:
            Processed results.
        """
        data["measure"] = "walk-in"
        data.loc[data["is_ambulance"], "measure"] = "ambulance"
        data = data.rename(columns={"arrivals": "value"})

        # summarise the results to make the create_agg steps quicker
        data = (
            data.groupby(
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
                dropna=False,
                as_index=False,
            )
            .agg({"value": "sum"})
            .fillna("unknown")
        )
        return data

    def specific_aggregations(self, model_results: pd.DataFrame) -> dict[str, pd.Series]:
        """Create other aggregations specific to the model type.

        Args:
            model_results: The results of a model run.

        Returns:
            Dictionary containing the specific aggregations.
        """
        return {
            "acuity": self.get_agg(model_results, "acuity"),
            "attendance_category": self.get_agg(model_results, "attendance_category"),
        }

    def calculate_avoided_activity(
        self, data: pd.DataFrame, data_resampled: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate the rows that have been avoided.

        Args:
            data: The data before the binomial thinning step.
            data_resampled: The data after the binomial thinning step.

        Returns:
            The data that was avoided in the binomial thinning step.
        """
        avoided = data["arrivals"] - data_resampled["arrivals"]
        data["arrivals"] = avoided
        return data

    def save_results(self, model_iteration: ModelIteration, path_fn: Callable[[str], str]) -> None:
        """Save the results of running the model.

        This method is used for saving the results of the model run to disk as a parquet file.
        It saves just the `rn` (row number) column and the `arrivals`, with the intention that
        you rejoin to the original data.

        Args:
            model_iteration: An instance of the ModelIteration class.
            path_fn: A function which takes the activity type and returns a path.
        """
        model_iteration.get_model_results().set_index(["rn"])[["arrivals"]].to_parquet(
            f"{path_fn('aae')}/0.parquet"
        )
        model_iteration.avoided_activity.set_index(["rn"])[["arrivals"]].to_parquet(
            f"{path_fn('aae_avoided')}/0.parquet"
        )
