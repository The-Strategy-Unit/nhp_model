"""
Inpatients Module

Implements the inpatients model.
"""

import json
from functools import partial
from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.helpers import inrange
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
        self, params: dict, data_path: str, hsa: Any, run_params: dict = None
    ) -> None:
        # call the parent init function
        super().__init__("ip", params, data_path, hsa, run_params)
        # load the theatres data
        self._load_theatres_data()
        # load the kh03 data
        self._load_kh03_data()

    def _get_data_mask(self) -> npt.ArrayLike:
        """get the data mask

        Some data is oversampled for modelling purposes, but should not be included in any counts.
        This method get's the appropriate data mask for the data

        :return: a boolean array the length of the data
        :rtype: npt.ArrayLike
        """
        return ~self.data["bedday_rows"].to_numpy()

    def _load_kh03_data(self):
        # load the kh03 data
        self._ward_groups = pd.concat(
            [
                pd.DataFrame(
                    {"ward_type": k, "ward_group": list(v.values())},
                    index=list(v.keys()),
                )
                for k, v in self.params["bed_occupancy"]["specialty_mapping"].items()
            ]
        )
        self._kh03_data = (
            pd.read_csv(
                f"{self._data_path}/kh03.csv",
                dtype={
                    "quarter": np.character,
                    "specialty_code": np.character,
                    "specialty_group": np.character,
                    "available": np.int64,
                    "occupied": np.int64,
                },
            )
            .merge(self._ward_groups, left_on="specialty_code", right_index=True)
            .groupby(["quarter", "ward_type", "ward_group"])[["available", "occupied"]]
            .agg(lambda x: sum(np.round(x).astype(int)))
        )
        # get the baseline data
        self._beds_baseline = self._bedday_summary(self.data, self.params["start_year"])

    def _load_theatres_data(self) -> None:
        """Load the Theatres data

        Loads the theatres data and stores in the private variable `self._theatres_data`
        """
        with open(f"{self._data_path}/theatres.json", "r", encoding="UTF-8") as tdf:
            self._theatres_data = json.load(tdf)

        fhs_baseline = pd.Series(
            self._theatres_data["four_hour_sessions"], name="four_hour_sessions"
        )

        baseline_params = pd.Series(
            {
                k: v["baseline"]
                for k, v in self.params["theatres"]["change_utilisation"].items()
            }
        )

        self._theatres_data["four_hour_sessions"] = fhs_baseline
        # sum the amount of procedures, by specialty,
        # then keep only the ones which appear in fhs_baseline
        self._procedures_baseline = self.data.groupby("tretspef")["has_procedure"].sum()
        self._procedures_baseline = self._procedures_baseline[
            self._procedures_baseline.index.isin(fhs_baseline.index)
        ]
        self._theatre_spells_baseline = (fhs_baseline / baseline_params).sum()

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

    def _get_data_counts(self, data: pd.DataFrame) -> npt.ArrayLike:
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
    ) -> Tuple[pd.DataFrame, npt.ArrayLike]:
        """Apply row resampling

        Called from within `model.activity_avoidance.ActivityAvoidance.apply_resampling`

        :param row_samples: [1xn] array, where n is the number of rows in `data`, containing the new
        values for `data["arrivals"]`
        :type row_samples: npt.ArrayLike
        :param data: the data that we want to update
        :type data: pd.DataFrame
        :return: the updated data
        :rtype: Tuple[pd.DataFrame, npt.ArrayLike]
        """
        data = data.loc[data.index.repeat(row_samples[0])].reset_index(drop=True)
        # filter out the "oversampled" rows from the counts
        mask = (~data["bedday_rows"]).to_numpy().astype(float)
        # return the altered data and the amount of admissions/beddays after resampling
        return (data, self._get_data_counts(data) * mask)

    def convert_step_counts(self, step_counts: dict) -> pd.DataFrame:
        """Convert the step counts

        :param step_counts: the step counts dictionary
        :type step_counts: dict
        :return: the step counts for uploading
        :rtype: dict
        """
        return self._convert_step_counts(step_counts, ["admissions", "beddays"])

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
                .losr_bads()
            )

    def _bedday_summary(self, data, year):
        """Summarise data to count how many beddays per quarter

        Calculated midnight bed occupancy per day, summarises to mean in quarter.

        :param data: the baseline data, or results of running the model
        :type data: pandas.DataFrame
        :param year: the year that the data represents
        :type year: int
        :return: a DataFrame containing a summary per quarter/ward group of maximum number of beds
            used in a quarter
        :rtype: pandas.DataFrame
        """
        return (
            data.query("(speldur > 0) & (classpat == '1')")
            .groupby(["admidate", "speldur", "mainspef"], as_index=False)
            .size()
            .rename(columns={"size": "nx"})
            .assign(day_n=lambda x: x.speldur.apply(np.arange))
            .explode("day_n")
            .groupby(["admidate", "day_n", "mainspef", "nx"], as_index=False)
            .size()
            .assign(
                date=lambda x: pd.to_datetime(x.admidate)
                + pd.to_timedelta(x.day_n, unit="d"),
                size=lambda x: x["nx"] * x["size"],
            )
            .merge(self._ward_groups, left_on="mainspef", right_index=True)
            .groupby(["date", "ward_type", "ward_group"], as_index=False)
            .agg({"size": sum})
            .assign(quarter=lambda x: x.date.dt.to_period("Q-MAR"))
            .query(f"quarter.dt.qyear == {year + 1}")
            .groupby(["quarter", "ward_type", "ward_group"], as_index=False)
            .agg({"size": np.mean})
            .assign(quarter=lambda x: x.quarter.astype(str).str[4:].str.lower())
        )

    def _bed_occupancy(self, data: pd.DataFrame, model_run: ModelRun) -> dict:
        """Calculate bed occupancy

        :param data: the baseline data, or the model results
        :type data: pandas.DataFrame
        :param bed_occupancy_params: the bed occupancy parameters from run_params
        :type bed_occupancy_params: dict
        :param model_run: the current model run
        :type model_run: int

        :returns: a dictionary of a named tuple to an integer. The named tuple contains the pod,
            measure, and the ward group. The value is the number of beds available
        :rtype: dict
        """
        bed_occupancy_params = model_run.run_params["bed_occupancy"]

        def create_dict_key(quarter, ward_type, ward_group):
            return frozenset(
                {
                    ("pod", "ip"),
                    ("measure", "day+night"),
                    ("quarter", quarter),
                    ("ward_type", ward_type),
                    ("ward_group", ward_group),
                }
            )

        if model_run.model_run == -1:
            return {
                create_dict_key(*k): np.round(v).astype(int).tolist()
                for k, v in self._kh03_data["available"].items()
            }

        target_bed_occupancy_rates = pd.Series(bed_occupancy_params["day+night"])
        target_bed_occupancy_rates.name = "target_occupancy_rate"

        beddays_model = (
            self._bedday_summary(data, self.params["start_year"])
            .rename(columns={"size": "model"})
            .merge(
                self._kh03_data["occupied"],
                left_on=["quarter", "ward_type", "ward_group"],
                right_index=True,
                how="outer",
            )
            .fillna(value={"occupied": 0})
            .set_index(["quarter", "ward_type", "ward_group"])
        )
        beddays_baseline = (
            self._beds_baseline.rename(columns={"size": "baseline"})
            .merge(target_bed_occupancy_rates, left_on="ward_group", right_index=True)
            .set_index(["quarter", "ward_type", "ward_group"])
        )

        beddays_results = (
            (
                (beddays_model["model"] * beddays_model["occupied"])
                / (
                    beddays_baseline["baseline"]
                    * beddays_baseline["target_occupancy_rate"]
                )
            )
            .fillna(0)
            .to_dict()
        )

        # return the results
        return {
            create_dict_key(*k): np.round(v).astype(int).tolist()
            for k, v in beddays_results.items()
        }

    def _theatres_available(
        self, model_results: pd.DataFrame, model_run: Model
    ) -> dict:
        """Calculate the theatres available

        :param model_results: a DataFrame containing the results of a model iteration
        :type model_results: pandas.DataFrame
        :param theatres_params: the theatres params from run_params
        :type theatres_params: dict
        :param model_run: the current model run
        :type model_run: int

        :returns: a dictionary of a named tuple to an integer. There are two set's of results that
            are returned: the number of theatres available, and the number of four hour sessions.
        :rtype: dict
        """
        # pylint: disable=too-many-locals
        # create the namedtuple types

        fhs_baseline = self._theatres_data["four_hour_sessions"]

        if model_run.model_run == -1:
            return {
                frozenset(
                    {
                        ("pod", "ip_theatres"),
                        ("measure", "four_hour_sessions"),
                        ("tretspef", k),
                    }
                ): v
                for k, v in fhs_baseline.items()
            }

        # sum the amount of procedures, by specialty,
        # then keep only the ones which appear in fhs_baseline
        future = (
            model_results[model_results["measure"] == "procedures"]
            .groupby("tretspef")["value"]
            .sum()
        )
        future = future[future.index.isin(fhs_baseline.index)]

        baseline = self._procedures_baseline

        fhs_future = (future / baseline * fhs_baseline).dropna()

        return {
            frozenset(
                zip(
                    ["pod", "measure", "tretspef"],
                    ["ip_theatres", "four_hour_sessions", k],
                )
            ): v
            for k, v in fhs_future.items()
        }

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

        bed_occupancy = self._bed_occupancy(model_results, model_run)

        model_results = (
            model_results[~model_results["bedday_rows"]]
            .groupby(
                [
                    "sitetret",
                    "age_group",
                    "sex",
                    "group",
                    "classpat",
                    "tretspef",
                ],
                # as_index = False
            )
            .agg({"rn": len, "has_procedure": sum, "speldur": sum})
            .assign(speldur=lambda x: x.speldur + x.rn)
            .rename(
                columns={
                    "rn": "admissions",
                    "has_procedure": "procedures",
                    "speldur": "beddays",
                }
            )
            .melt(ignore_index=False, var_name="measure")
            .reset_index()
        )

        model_results["pod"] = "ip_" + model_results["group"] + "_admission"
        # quick dq fix: convert any "non-elective" daycases to "elective"
        model_results.loc[
            model_results["classpat"].isin(["2", "3"]), "pod"
        ] = "ip_elective_daycase"

        # handle the outpatients rows
        op_rows = model_results.classpat == "-1"
        # filter out op_rows we don't care for
        model_results = model_results[
            ~op_rows | (model_results.measure == "admissions")
        ]
        # update the columns
        model_results.loc[op_rows, "measure"] = "attendances"
        model_results.loc[op_rows, "pod"] = "op_procedure"

        theatres_available = self._theatres_available(
            model_results.loc[~op_rows],
            model_run,
        )

        agg = partial(self._create_agg, model_results)
        return (
            agg,
            {
                **agg(["sex", "tretspef"]),
                "bed_occupancy": bed_occupancy,
                "theatres_available": theatres_available,
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
            .fillna("NULL")
            # .loc[self.data["rn"]]
            .rename(None)
        )
        self.data.set_index(selected_strategy, inplace=True)

    def _generate_losr_df(self):
        params = self._model_run.params["efficiencies"]["ip"]
        run_params = self._model_run.run_params["efficiencies"]["ip"]
        losr = pd.DataFrame.from_dict(params, orient="index")
        losr["losr_f"] = [run_params[i] for i in losr.index]
        self.losr = losr

    def _update(self, i, new):
        mask = ~self.data.loc[i, "bedday_rows"]
        pre_los = self.data.loc[i, "speldur"]
        self.data.loc[i, "speldur"] = new
        change_los = (
            ((self.data.loc[i, "speldur"] - pre_los) * mask)
            .groupby(level=0)
            .sum()
            .astype(int)
        ).to_dict()

        for k in change_los.keys():
            self.step_counts[("efficiencies", k)] = np.array([0, change_los[k]])

        return self

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
        return self._update(i, new)

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
        return self._update(i, new)

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
        return self._update(i, new)

    def losr_bads(self) -> None:
        """
        Length of Stay Reduction: British Association of Day Surgery

        This will swap rows between elective admissions and daycases into either daycases or
        outpatients, based on the given parameter values. We have a baseline target rate, this is
        the rate in the baseline that rows are of the given target type (i.e. either daycase,
        outpatients, daycase or outpatients). Our interval will alter the target_rate by setting
        some rows which are not the target type to be the target type.

        Rows that are converted to daycase have the patient classification set to 2. Rows that are
        converted to outpatients have a patient classification of -1 (these rows need to be filtered
        out of the inpatients results and added to the outpatients results).

        Rows that are modelled away from elective care have the length of stay fixed to 0 days.
        """
        #
        losr = self.losr
        data = self.data
        rng = self._model_run.rng

        losr = losr[losr.type.str[:4] == "bads"]

        i = losr.index

        factor = data.merge(
            losr["losr_f"],
            left_index=True,
            right_index=True,
        )["losr_f"]

        beddays_before = (
            (data.loc[i, "speldur"] * ~data.loc[i, "bedday_rows"])
            .groupby(level=0)
            .sum()
        )

        dont_change_classpat = rng.binomial(1, factor).astype(bool)
        data.loc[i, "speldur"] *= dont_change_classpat

        # change the classpat column
        data.loc[i, "classpat"] = np.where(
            dont_change_classpat,
            data.loc[i, "classpat"],
            np.where(data.loc[i].index.str[5:12] == "daycase", "2", "-1"),
        )

        step_counts_admissions = -(
            ((data.loc[i, "classpat"] == "-1") * ~data.loc[i, "bedday_rows"])
            .groupby(level=0)
            .sum()
            .astype(int)
        )
        step_counts_beddays = (
            (
                (
                    (data.loc[i, "speldur"] * ~data.loc[i, "bedday_rows"])
                    .groupby(level=0)
                    .sum()
                    - beddays_before
                    + step_counts_admissions
                )
            )
            .astype(int)
            .to_dict()
        )

        step_counts_admissions = step_counts_admissions.to_dict()
        for k in step_counts_admissions.keys():
            self.step_counts[("efficiencies", k)] = np.array(
                [step_counts_admissions[k], step_counts_beddays[k]]
            )

        return self
