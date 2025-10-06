"""Outpatients Module.

Implements the Outpatients model.
"""

from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd

from nhp.model.data import Data
from nhp.model.model import Model
from nhp.model.model_iteration import ModelIteration


class OutpatientsModel(Model):
    """Outpatients Model.

    Implementation of the Model for Outpatient attendances.

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
        """Initialise the Outpatients Model.

        Args:
            params: The parameters to use.
            data: A method to create a Data instance.
            hsa: Health Status Adjustment object. Defaults to None.
            run_params: The run parameters to use. Defaults to None.
            save_full_model_results: Whether to save full model results. Defaults to False.
        """
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

    def get_data_counts(self, data: pd.DataFrame) -> np.ndarray:
        """Get row counts of data.

        Args:
            data: The data to get the counts of.

        Returns:
            The counts of the data, required for activity avoidance steps.
        """
        return data[["attendances", "tele_attendances"]].to_numpy().astype(float).transpose()

    def _load_strategies(self, data_loader: Data) -> None:
        data = self.data.set_index("rn")

        activity_avoidance = pd.concat(
            [
                "followup_reduction_" + data[~data["is_first"] & ~data["has_procedures"]]["type"],
                "consultant_to_consultant_reduction_" + data[data["is_cons_cons_ref"]]["type"],
                "gp_referred_first_attendance_reduction_"
                + data[data["is_gp_ref"] & data["is_first"]]["type"],
            ]
        )
        efficiencies: pd.Series = pd.concat(  # type: ignore
            ["convert_to_tele_" + data[~data["has_procedures"]]["type"]]
        )

        self.strategies: dict[str, pd.DataFrame] = {
            k: v.rename("strategy").to_frame().assign(sample_rate=1)
            for k, v in {
                "activity_avoidance": activity_avoidance,
                "efficiencies": efficiencies,
            }.items()
        }

    @staticmethod
    def _convert_to_tele(
        data: pd.DataFrame,
        model_iteration: ModelIteration,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert attendances to tele-attendances.

        Args:
            data: The DataFrame that we are updating.
            model_iteration: The model iteration containing the RNG and run parameters.

        Returns:
            A tuple containing the updated data and the updated step counts.
        """
        # TODO: we need to make sure efficiences contains convert to tele keys
        rng = model_iteration.rng
        params = model_iteration.run_params["efficiencies"]["op"]
        strategies = model_iteration.model.strategies["efficiencies"]
        # make sure to take the complement of the parameter
        factor = 1 - data["rn"].map(strategies["strategy"].map(params)).fillna(1)
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

    def apply_resampling(self, row_samples: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
        """Apply row resampling.

        Called from within `model.activity_resampling.ActivityResampling.apply_resampling`.

        Args:
            row_samples: [2xn] array, where n is the number of rows in `data`, containing the new
                values for `data["attendances"]` and `data["tele_attendances"]`.
            data: The data that we want to update.

        Returns:
            The updated data.
        """
        data["attendances"] = row_samples[0]
        data["tele_attendances"] = row_samples[1]
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
            Tuple containing the updated data and step counts.
        """
        data, step_counts = self._convert_to_tele(data, model_iteration)
        return data, step_counts

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
        avoided = (
            data[["attendances", "tele_attendances"]]
            - data_resampled[["attendances", "tele_attendances"]]
        )
        data[["attendances", "tele_attendances"]] = avoided
        return data

    @staticmethod
    def process_results(data: pd.DataFrame) -> pd.DataFrame:
        """Process the data into a format suitable for aggregation in results files.

        Args:
            data: Data to be processed. Format should be similar to Model.data.

        Returns:
            Processed results.
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

    def specific_aggregations(self, model_results: pd.DataFrame) -> dict[str, pd.Series]:
        """Create other aggregations specific to the model type.

        Args:
            model_results: The results of a model run.

        Returns:
            Dictionary containing the specific aggregations.
        """
        return {
            "sex+tretspef_grouped": self.get_agg(model_results, "sex", "tretspef_grouped"),
            "tretspef": self.get_agg(model_results, "tretspef"),
        }

    def save_results(self, model_iteration: ModelIteration, path_fn: Callable[[str], str]) -> None:
        """Save the results of running the model.

        This method is used for saving the results of the model run to disk as a parquet file.
        It saves just the `rn` (row number) column and the `attendances` and `tele_attendances`
        columns, with the intention that you rejoin to the original data.

        Args:
            model_iteration: An instance of the ModelIteration class.
            path_fn: A function which takes the activity type and returns a path.
        """
        model_iteration.get_model_results().set_index(["rn"])[
            ["attendances", "tele_attendances"]
        ].to_parquet(f"{path_fn('op')}/0.parquet")

        model_iteration.avoided_activity.set_index(["rn"])[
            ["attendances", "tele_attendances"]
        ].to_parquet(f"{path_fn('op_avoided')}/0.parquet")
