"""Inpatients Module.

Implements the inpatients model.
"""

from collections import defaultdict
from typing import Any, Callable, Self, Tuple

import numpy as np
import pandas as pd

from nhp.model.data import Data
from nhp.model.model import Model
from nhp.model.model_iteration import ModelIteration


class InpatientsModel(Model):
    """Inpatients Model.

    Implementation of the Model for Inpatient admissions.

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
        """Initialise the Inpatients Model.

        Args:
            params: The parameters to use.
            data: A method to create a Data instance.
            hsa: Health Status Adjustment object. Defaults to None.
            run_params: The run parameters to use. Defaults to None.
            save_full_model_results: Whether to save full model results. Defaults to False.
        """
        # call the parent init function
        super().__init__(
            "ip",
            ["admissions", "beddays"],
            params,
            data,
            hsa,
            run_params,
            save_full_model_results,
        )

    def _get_data(self, data_loader: Data) -> pd.DataFrame:
        return data_loader.get_ip()

    def _load_strategies(self, data_loader: Data) -> None:
        """Load a set of strategies."""

        def filter_valid(strategy_type: str, strats: pd.DataFrame) -> pd.DataFrame:
            strats = strats.set_index(["rn"])
            # get the valid set of valid strategies from the params
            valid_strats = self.params[strategy_type]["ip"].keys()
            # subset the strategies
            return strats[strats.iloc[:, 0].isin(valid_strats)].rename(
                columns={strats.columns[0]: "strategy"}
            )

        def append_general_los(i, j):
            j = f"general_los_reduction_{j}"
            if j not in self.params["efficiencies"]["ip"]:
                return None

            return (
                self.data.query(f"admimeth.str.startswith('{i}') & (speldur > 0) & classpat == '1'")
                .set_index("rn")[[]]
                .drop_duplicates()
                .drop_duplicates()
                .merge(
                    self.strategies["efficiencies"],
                    how="left",
                    left_index=True,
                    right_index=True,
                    indicator=True,
                )
                .query("_merge != 'both'")
                .drop(columns="_merge")
                .fillna({"strategy": j, "sample_rate": 1.0})
            )

        self.strategies = {
            k: filter_valid(k, v) for k, v in data_loader.get_ip_strategies().items()
        }

        # set admissions without a strategy selected to general_los_reduction_X
        # where applicable
        self.strategies["efficiencies"] = pd.concat(
            [self.strategies["efficiencies"]]
            + [
                # see https://github.com/astral-sh/ty/issues/247
                append_general_los(*x)  # ty: ignore [missing-argument]
                for x in [("1", "elective"), ("2", "emergency")]
            ]
        )

    def get_data_counts(self, data: pd.DataFrame) -> np.ndarray:
        """Get row counts of data.

        Args:
            data: The data to get the counts of.

        Returns:
            The counts of the data, required for activity avoidance steps.
        """
        return np.array([np.ones_like(data["rn"]), (1 + data["speldur"]).to_numpy()]).astype(float)

    def apply_resampling(self, row_samples: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
        """Apply row resampling.

        Called from within `model.activity_resampling.ActivityResampling.apply_resampling`.

        Args:
            row_samples: [1xn] array, where n is the number of rows in `data`, containing the new
                values for admissions.
            data: The data that we want to update.

        Returns:
            The updated data.
        """
        return data.loc[data.index.repeat(row_samples[0])].reset_index(drop=True)

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
        # skip if there are no efficiencies
        if model_iteration.model.strategies["efficiencies"].empty:
            return data, None

        efficiencies = (
            InpatientEfficiencies(data, model_iteration)
            .losr_all()
            .losr_sdec()
            .losr_preop()
            .losr_day_procedures("day_procedures_daycase")
            .losr_day_procedures("day_procedures_outpatients")
        )

        return efficiencies.data, efficiencies.get_step_counts()

    @staticmethod
    def process_results(data: pd.DataFrame) -> pd.DataFrame:
        """Process the data into a format suitable for aggregation in results files.

        Args:
            data: Data to be processed. Format should be similar to Model.data.

        Returns:
            Processed results.
        """
        # handle the type conversions: change the pod's
        data.loc[data["classpat"] == "-2", "pod"] = "ip_elective_daycase"
        data.loc[data["classpat"] == "-1", "pod"] = "op_procedure"
        data.loc[data["classpat"] == "-3", "pod"] = "aae_type-05"

        los_groups = defaultdict(
            lambda: "22+ days",
            {
                0: "0 days",
                1: "1 day",
                2: "2 days",
                3: "3 days",
                **{i: "4-7 days" for i in range(4, 8)},
                **{i: "8-14 days" for i in range(8, 15)},
                **{i: "15-21 days" for i in range(15, 22)},
            },
        )

        data = (
            data.assign(
                los_group=lambda x: x["speldur"].map(los_groups),
                admissions=1,
                beddays=lambda x: x["speldur"] + 1,
                procedures=lambda x: x["has_procedure"],
            )
            .groupby(
                [
                    "sitetret",
                    "age",
                    "age_group",
                    "sex",
                    "pod",
                    "tretspef",
                    "tretspef_grouped",
                    "los_group",
                    "maternity_delivery_in_spell",
                ],
                dropna=False,
            )[["admissions", "beddays", "procedures"]]
            .sum()
        )
        # handle any outpatients/sdec rows
        data.loc[
            data.index.get_level_values("pod").isin(["op_procedure", "aae_type-05"]),
            ["beddays", "procedures"],
        ] = 0
        data = data.melt(ignore_index=False, var_name="measure")

        # remove any row where the measure value is 0
        data = data[data["value"] > 0].reset_index()

        # handle op_procedure column values
        op_ix = data["pod"] == "op_procedure"
        data.loc[op_ix, "measure"] = "attendances"
        data.loc[op_ix, "los_group"] = None

        # handle sdec column values
        sdec_ix = data["pod"] == "aae_type-05"
        data.loc[sdec_ix, "measure"] = "walk-in"
        data.loc[sdec_ix, "tretspef"] = "Other"
        data.loc[sdec_ix, "tretspef_grouped"] = "Other"
        data.loc[sdec_ix, "los_group"] = None

        # reduce any duplicated rows
        data = data.groupby(
            data.drop(columns="value").columns.tolist(), as_index=False, dropna=False
        ).sum()

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
            "tretspef+los_group": self.get_agg(model_results, "tretspef", "los_group"),
            "delivery_episode_in_spell": self.get_agg(
                model_results[model_results["maternity_delivery_in_spell"]]
            ),
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
        diffs = data["rn"].value_counts() - data_resampled["rn"].value_counts()
        rows_avoided = diffs.fillna(data["rn"].value_counts()).sort_index()
        rows_avoided = pd.DataFrame(
            np.sort(data["rn"].unique()).repeat(rows_avoided), columns=["rn"]
        )
        return rows_avoided.merge(data.drop_duplicates(), how="left", on="rn")

    def save_results(self, model_iteration: ModelIteration, path_fn: Callable[[str], str]) -> None:
        """Save the results of running the model.

        Args:
            model_iteration: An instance of the ModelIteration class.
            path_fn: A function which takes the activity type and returns a path.
        """
        model_results = model_iteration.get_model_results()

        # save the op converted rows
        self._save_results_get_op_converted(model_results).to_parquet(
            f"{path_fn('op_conversion')}/0.parquet"
        )
        # save the sdec converted rows
        self._save_results_get_sdec_converted(model_results).to_parquet(
            f"{path_fn('sdec_conversion')}/0.parquet"
        )
        # save the ip rows
        self._save_results_get_ip_rows(model_results).to_parquet(f"{path_fn('ip')}/0.parquet")
        # save the avoided activity
        model_iteration.avoided_activity[["rn", "speldur", "classpat"]].to_parquet(
            f"{path_fn('ip_avoided')}/0.parquet"
        )

    def _save_results_get_op_converted(self, model_results: pd.DataFrame) -> pd.DataFrame:
        ix = model_results["classpat"] == "-1"
        return (
            model_results[ix]
            .groupby(["age", "sex", "tretspef", "tretspef_grouped", "sitetret"])
            .size()
            .to_frame("attendances")
            .assign(tele_attendances=0)
            .reset_index()
        )

    def _save_results_get_sdec_converted(self, model_results: pd.DataFrame) -> pd.DataFrame:
        ix = model_results["classpat"] == "-3"
        return (
            model_results[ix]
            .groupby(["age", "sex", "sitetret"])
            .size()
            .to_frame("arrivals")
            .assign(
                aedepttype="05",
                attendance_category="1",
                acuity="standard",
                group="walk-in",
            )
            .reset_index()
        )

    def _save_results_get_ip_rows(self, model_results: pd.DataFrame) -> pd.DataFrame:
        # update the converted daycase rows
        model_results.loc[model_results["classpat"] == "-2", "classpat"] = "2"

        ix = ~model_results["classpat"].str.startswith("-")
        return model_results.loc[ix, ["rn", "speldur", "classpat"]]


class InpatientEfficiencies:
    """Apply the Inpatient Efficiency Strategies."""

    def __init__(self, data: pd.DataFrame, model_iteration: ModelIteration):
        """Initialise the InpatientEfficiencies.

        Args:
            data: The data which we are updating.
            model_iteration: A ModelIteration instance for the current model run.
        """
        self.data = data
        self._model_iteration = model_iteration
        self._select_single_strategy()
        self._generate_losr_df()
        self.speldur_before = self.data["speldur"].copy()

    @property
    def strategies(self):
        """Get the efficiencies strategies."""
        return self._model_iteration.model.strategies["efficiencies"]

    def _select_single_strategy(self) -> None:
        rng = self._model_iteration.rng
        selected_strategy = (
            self.data["rn"]
            .map(
                self.strategies["strategy"]
                .sample(frac=1, random_state=rng.bit_generator)
                .groupby(level=0)
                .head(1),
                na_action="ignore",
            )
            .rename(None)
        )
        # assign the selected strategies
        self.data = self.data.set_index(selected_strategy)

    def _generate_losr_df(self) -> None:
        params = self._model_iteration.params["efficiencies"]["ip"]
        run_params = self._model_iteration.run_params["efficiencies"]["ip"]
        losr = pd.DataFrame.from_dict(params, orient="index")
        losr["losr_f"] = [run_params[i] for i in losr.index]
        self.losr = losr

    def losr_all(self) -> Self:
        """Length of Stay Reduction: All.

        Reduces all rows length of stay by sampling from a binomial distribution, using the current
        length of stay as the value for n, and the length of stay reduction factor for that strategy
        as the value for p. This will update the los to be a value between 0 and the original los.
        """
        losr = self.losr
        data = self.data
        rng = self._model_iteration.rng

        i = losr.index[(losr.type == "all") & (losr.index.isin(data.index))]

        if i.empty:
            return self

        new = rng.binomial(data.loc[i, "speldur"], losr.loc[data.loc[i].index, "losr_f"])

        self.data.loc[i, "speldur"] = new.astype("int32")

        return self

    def losr_sdec(self) -> Self:
        """Length of Stay Reduction: SDEC reduction.

        Converts IP activity to SDEC attendance for a given percentage of rows.
        """
        losr = self.losr
        data = self.data
        rng = self._model_iteration.rng

        i = losr.index[(losr.type == "sdec") & (losr.index.isin(data.index))]

        if i.empty:
            return self

        rnd_choice = np.array(rng.binomial(1, losr.loc[data.loc[i].index, "losr_f"])).astype(
            "int32"
        )

        self.data.loc[i, "classpat"] = np.where(rnd_choice == 0, "-3", "1")
        self.data.loc[i, "speldur"] = self.data.loc[i, "speldur"] * rnd_choice

        return self

    def losr_preop(self) -> Self:
        """Length of Stay Reduction: Pre-op reduction.

        Updates the length of stay to by removing 1 or 2 days for a given percentage of rows
        """
        losr = self.losr
        data = self.data
        rng = self._model_iteration.rng

        i = losr.index[(losr.type == "pre-op") & (losr.index.isin(data.index))]

        if i.empty:
            return self

        new = data.loc[i, "speldur"] - (
            rng.binomial(1, 1 - losr.loc[data.loc[i].index, "losr_f"])
            * losr.loc[data.loc[i].index, "pre-op_days"]
        )
        self.data.loc[i, "speldur"] = new.astype("int32")

        return self

    def losr_day_procedures(self, day_procedure_type: str) -> Self:
        """Length of Stay Reduction: Day Procedures.

        This will swap rows between elective admissions and daycases into either daycases or
        outpatients, based on the given parameter values.

        Rows that are converted to daycase have the patient classification set to 2. Rows that are
        converted to outpatients have a patient classification of -1 (these rows need to be filtered
        out of the inpatients results and added to the outpatients results).

        Rows that are modelled away from elective care have the length of stay fixed to 0 days.
        """
        losr = self.losr
        data = self.data
        rng = self._model_iteration.rng

        losr = losr[losr.type == day_procedure_type]

        i = losr.index

        factor = data.merge(
            losr["losr_f"],
            left_index=True,
            right_index=True,
        )["losr_f"]

        dont_change_classpat = rng.binomial(1, factor).astype(bool)
        data.loc[i, "speldur"] *= dont_change_classpat

        # change the classpat column
        data.loc[i, "classpat"] = np.where(
            dont_change_classpat,
            data.loc[i, "classpat"],
            "-2" if day_procedure_type == "day_procedures_daycase" else "-1",
        )

        return self

    def get_step_counts(self) -> pd.DataFrame:
        """Updates the step counts object.

        After running the efficiencies, update the model runs step counts object.
        """
        return (
            self.data
            # handle the changes of activity type
            .assign(admissions=lambda x: np.where(x["classpat"].isin(["-1", "-2", "-3"]), -1, 0))
            .assign(
                beddays=lambda x: x["speldur"]
                - self.speldur_before
                # any admissions that are converted to outpatients will reduce 1 bedday per
                # admission this column is negative values, so we need to add in order to subtract
                + x["admissions"]
            )
            .loc[self.data.index.notna()]
            .reset_index()
            .rename(columns={"index": "strategy"})
            .groupby(["pod", "sitetret", "strategy"], as_index=False)[["admissions", "beddays"]]
            .sum()
            .assign(change_factor="efficiencies")
        )
