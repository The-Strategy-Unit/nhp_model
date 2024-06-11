"""
Inpatients Module

Implements the inpatients model.
"""

from functools import partial
from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.model import Model
from model.model_run import ModelRun


class InpatientsModel(Model):
    """Inpatients Model

    :param params: the parameters to run the model with
    :type params: dict
    :param data_path: the path to where the data files live
    :type data_path: str

    Inherits from the Model class.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        params: dict,
        data_path: str,
        hsa: Any = None,
        run_params: dict = None,
        save_full_model_results: bool = False,
    ) -> None:
        # call the parent init function
        super().__init__(
            "ip",
            ["admissions", "beddays"],
            params,
            data_path,
            hsa,
            run_params,
            save_full_model_results,
        )

    def _add_pod_to_data(self) -> None:
        """Adds the POD column to data"""
        self.data["pod"] = "ip_" + self.data["group"] + "_admission"
        self.data.loc[self.data["classpat"].isin(["2", "3"]), "pod"] = (
            "ip_elective_daycase"
        )

    def _load_strategies(self) -> None:
        """Load a set of strategies"""

        def load_fn(strategy_type):
            # load the file
            strats = self._load_parquet(f"ip_{strategy_type}_strategies").set_index(
                ["rn"]
            )
            # get the valid set of valid strategies from the params
            valid_strats = self.params[strategy_type]["ip"].keys()
            # subset the strategies
            return strats[strats.iloc[:, 0].isin(valid_strats)].rename(
                columns={strats.columns[0]: "strategy"}
            )

        self.strategies = {
            k: load_fn(k) for k in ["activity_avoidance", "efficiencies"]
        }

        def append_general_los(i, j):
            j = f"general_los_reduction_{j}"
            if j not in self.params["efficiencies"]["ip"]:
                return None

            return (
                self.data.query(
                    f"admimeth.str.startswith('{i}') & (speldur > 0) & classpat == '1'"
                )
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

        # set admissions without a strategy selected to general_los_reduction_X
        # where applicable
        self.strategies["efficiencies"] = pd.concat(
            [self.strategies["efficiencies"]]
            + [append_general_los(*x) for x in [("1", "elective"), ("2", "emergency")]]
        )

    def get_data_counts(self, data: pd.DataFrame) -> npt.ArrayLike:
        """Get row counts of data

        :param data: the data to get the counts of
        :type data: pd.DataFrame
        :return: the counts of the data, required for activity avoidance steps
        :rtype: npt.ArrayLike
        """
        return np.array(
            [np.ones_like(data["rn"]), (1 + data["speldur"]).to_numpy()]
        ).astype(float)

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
        return data.loc[data.index.repeat(row_samples[0])].reset_index(drop=True)

    def get_future_from_row_samples(self, row_samples):
        """Get the future counts from row samples

        Called from within `model.activity_avoidance.ActivityAvoidance.apply_resampling`
        """
        return row_samples[0] * self.baseline_counts

    def efficiencies(self, model_run: ModelRun) -> None:
        """Run the efficiencies steps of the model

        :param model_run: an instance of the ModelRun class
        :type model_run: model.model_run.ModelRun
        """
        # skip if there are no efficiencies
        if not model_run.model.strategies["efficiencies"].empty:
            (
                InpatientEfficiencies(model_run)
                .losr_all()
                .losr_aec()
                .losr_preop()
                .losr_day_procedures("day_procedures_daycase")
                .losr_day_procedures("day_procedures_outpatients")
                .update_step_counts()
            )

    def aggregate(self, model_run: ModelRun) -> Tuple[Callable, dict]:
        """Aggregate the model results

        Can also be used to aggregate the baseline data by passing in a `ModelRun` with
        the `model_run` argument set `-1`.

        :param model_run: an instance of the `ModelRun` class
        :type model_run: model.model_run.ModelRun

        :returns: a dictionary containing the different aggregations of this data
        :rtype: dict
        """
        # get the run params: use principal run for baseline
        model_results = model_run.get_model_results()

        model_results = (
            model_results.assign(
                los_group=lambda x: np.where(
                    x["speldur"] == 0,
                    "0-day",
                    np.where(
                        x["speldur"] <= 7,
                        "1-7 days",
                        np.where(
                            x["speldur"] <= 14,
                            "8-14 days",
                            np.where(x["speldur"] <= 21, "15-21 days", "22+ days"),
                        ),
                    ),
                ),
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
                    "group",
                    "pod",
                    "classpat",
                    "tretspef",
                    "tretspef_raw",
                    "los_group",
                ],
                dropna=False,
            )[["admissions", "beddays", "procedures"]]
            .sum()
            .melt(ignore_index=False, var_name="measure")
            .reset_index()
        )

        # quick dq fix: convert any "non-elective" daycases to "elective"
        model_results.loc[model_results["classpat"].isin(["-2", "2", "3"]), "pod"] = (
            "ip_elective_daycase"
        )

        # handle the outpatients rows
        op_rows = model_results.classpat == "-1"
        # filter out op_rows we don't care for
        model_results = model_results[
            ~op_rows | (model_results.measure == "admissions")
        ]
        # update the columns
        model_results.loc[op_rows, "measure"] = "attendances"
        model_results.loc[op_rows, "pod"] = "op_procedure"

        agg = partial(self._create_agg, model_results)
        return (
            agg,
            {
                **agg(["sex", "tretspef"]),
                **agg(["tretspef_raw", "los_group"]),
            },
        )

    def save_results(self, model_run: ModelRun, path_fn: Callable[[str], str]) -> None:
        """Save the results of running the model

        :param model_run: an instance of the `ModelRun` class
        :type model_run: model.model_run.ModelRun
        :param path_fn: a function which takes the activity type and returns a path
        :type path_fn: Callable[[str], str]
        """
        model_results = model_run.get_model_results()
        ip_op_row_ix = model_results["classpat"] == "-1"
        # save the op converted rows
        model_results[ip_op_row_ix].groupby(["age", "sex", "tretspef"]).size().to_frame(
            "attendances"
        ).assign(tele_attendances=0).reset_index().to_parquet(
            f"{path_fn('op_conversion')}/0.parquet"
        )
        # remove the op converted rows
        model_results.loc[~ip_op_row_ix, ["speldur", "classpat"]].to_parquet(
            f"{path_fn('ip')}/0.parquet"
        )


class InpatientEfficiencies:
    """Apply the Inpatient Efficiency Strategies"""

    def __init__(self, model_run: ModelRun):
        self._model_run = model_run
        self.step_counts = model_run.step_counts
        self._select_single_strategy()
        self._generate_losr_df()
        self.speldur_before = self.data["speldur"].copy()

    @property
    def data(self):
        """get the model runs data"""
        return self._model_run.data

    @property
    def strategies(self):
        """get the efficiencies strategies"""
        return self._model_run.model.strategies["efficiencies"]

    def _select_single_strategy(self):
        rng = self._model_run.rng
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
        self.data.set_index(selected_strategy, inplace=True)

    def _generate_losr_df(self):
        params = self._model_run.params["efficiencies"]["ip"]
        run_params = self._model_run.run_params["efficiencies"]["ip"]
        losr = pd.DataFrame.from_dict(params, orient="index")
        losr["losr_f"] = [run_params[i] for i in losr.index]
        self.losr = losr

    def losr_all(self):
        """Length of Stay Reduction: All

        Reduces all rows length of stay by sampling from a binomial distribution, using the current
        length of stay as the value for n, and the length of stay reduction factor for that strategy
        as the value for p. This will update the los to be a value between 0 and the original los.
        """
        losr = self.losr
        data = self.data
        rng = self._model_run.rng

        i = losr.index[(losr.type == "all") & (losr.index.isin(data.index))]

        if i.empty:
            return self

        new = rng.binomial(
            data.loc[i, "speldur"], losr.loc[data.loc[i].index, "losr_f"]
        )

        self.data.loc[i, "speldur"] = new

        return self

    def losr_aec(self):
        """
        Length of Stay Reduction: AEC reduction

        Updates the length of stay to 0 for a given percentage of rows.
        """
        losr = self.losr
        data = self.data
        rng = self._model_run.rng

        i = losr.index[(losr.type == "aec") & (losr.index.isin(data.index))]

        if i.empty:
            return self

        new = data.loc[i, "speldur"] * rng.binomial(
            1, losr.loc[data.loc[i].index, "losr_f"]
        )

        self.data.loc[i, "speldur"] = new

        return self

    def losr_preop(self):
        """
        Length of Stay Reduction: Pre-op reduction

        Updates the length of stay to by removing 1 or 2 days for a given percentage of rows
        """
        losr = self.losr
        data = self.data
        rng = self._model_run.rng

        i = losr.index[(losr.type == "pre-op") & (losr.index.isin(data.index))]

        if i.empty:
            return self

        new = data.loc[i, "speldur"] - (
            rng.binomial(1, 1 - losr.loc[data.loc[i].index, "losr_f"])
            * losr.loc[data.loc[i].index, "pre-op_days"]
        )
        self.data.loc[i, "speldur"] = new

        return self

    def losr_day_procedures(self, day_procedure_type: str) -> None:
        """
        Length of Stay Reduction: Day Procedures

        This will swap rows between elective admissions and daycases into either daycases or
        outpatients, based on the given parameter values.

        Rows that are converted to daycase have the patient classification set to 2. Rows that are
        converted to outpatients have a patient classification of -1 (these rows need to be filtered
        out of the inpatients results and added to the outpatients results).

        Rows that are modelled away from elective care have the length of stay fixed to 0 days.
        """
        #
        losr = self.losr
        data = self.data
        rng = self._model_run.rng

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

    def update_step_counts(self):
        """Updates the step counts object

        After running the efficiencies, update the model runs step counts object.
        """
        df = (
            self.data
            # handle the changes of activity type
            .assign(
                admissions=lambda x: np.where(x["classpat"].isin(["-1", "-2"]), -1, 0)
            )
            .assign(
                beddays=lambda x: x["speldur"]
                - self.speldur_before
                # any admissions that are converted to outpatients will reduce 1 bedday per
                # admission this column is negative values, so we need to add in order to subtract
                + x["admissions"]
            )[["rn", "admissions", "beddays"]]
            .reset_index()
            .rename(columns={"index": "strategy"})
            .groupby(["rn", "strategy"], as_index=False)
            .sum()
            .melt(["rn", "strategy"], var_name="measure")
            .pivot(index=["measure", "rn"], columns="strategy", values="value")
        )

        model_run = self._model_run
        rn = pd.Index(model_run.model.data["rn"])
        for s in df.columns:
            model_run.step_counts[("efficiencies", s)] = np.array(
                [
                    rn.map(df.loc[(m, slice(None))][s]).fillna(0.0).to_numpy()
                    for m in df.index.levels[0]
                ]
            )

        return self
