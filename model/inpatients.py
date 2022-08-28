"""
Inpatients Module

Implements the inpatients model.
"""
import json
from collections import defaultdict, namedtuple
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from model.helpers import inrange
from model.model import Model


class InpatientsAdmissionsCounter:
    def __init__(self, data, rng):
        self._data = data
        self.rng = rng
        self.admissions = np.ones_like(data.index)
        self._beddays = (data["speldur"] + 1).astype(int)
        self.step_counts = {
            ("baseline", "-"): {
                "admissions": len(self.admissions),
                "beddays": sum(self._beddays),
            }
        }

    def _update_step_counts(self, new_admissions, name, split_by=None):
        diff = new_admissions - self.admissions

        if split_by == None:
            self.step_counts[(name, "-")] = {
                "admissions": sum(diff),
                "beddays": sum(diff * self._beddays),
            }
        else:
            counts = (
                pd.DataFrame({"admissions": diff, "beddays": diff * self._beddays})
                .groupby(split_by)
                .sum()
                .to_dict(orient="index")
            )

            self.step_counts.update({(name, k): v for k, v in counts.items()})

        self.admissions = new_admissions
        return self

    def poisson_step(self, factor, name, split_by=None):
        return self._update_step_counts(
            self.rng.poisson(factor * self.admissions), name, split_by
        )

    def binomial_step(self, factor, name, split_by=None):
        return self._update_step_counts(
            self.rng.binomial(n=self.admissions, p=factor), name, split_by
        )

    def get_data(self):
        return self._data.loc[self._data.index.repeat(self.admissions)].reset_index(
            drop=True
        )


class InpatientsModel(Model):
    """Inpatients Model

    :param params: the parameters to run the model with
    :type params: dict
    :param data_path: the path to where the data files live
    :type data_path: str

    Inherits from the Model class.
    """

    def __init__(self, params: list, data_path: str) -> None:
        # call the parent init function
        super().__init__(
            "ip",
            params,
            data_path,
            columns_to_load=[
                "rn",
                "speldur",
                "age",
                "sex",
                "admimeth",
                "classpat",
                "mainspef",
                "tretspef",
                "hsagrp",
                "has_procedure",
            ],
        )
        # load the strategies, store each strategy file as a separate entry in a dictionary
        self._strategies = {
            x: self._load_parquet(f"ip_{x}_strategies").set_index(["rn"])
            for x in ["admission_avoidance", "los_reduction"]
        }
        # load the theatres data
        self._load_theatres_data()
        # load the kh03 data
        self._load_kh03_data()

    def _load_kh03_data(self):
        # load the kh03 data
        self._ga_ward_groups = pd.Series(
            self.params["bed_occupancy"]["specialty_mapping"]["General and Acute"],
            name="ward_group",
        )
        self._kh03_data = (
            pd.read_csv(
                f"{self._data_path}/kh03.csv",
                dtype={
                    "specialty_code": np.character,
                    "specialty_group": np.character,
                    "available": np.float64,
                    "occupied": np.float64,
                },
            )
            .merge(self._ga_ward_groups, left_on="specialty_code", right_index=True)
            .groupby(["ward_group"])
            .agg("sum")
        )
        # get the baseline data
        self._beds_baseline = (
            self.data[self.data.classpat.isin(["1", "4"])]
            .merge(self._ga_ward_groups, left_on="mainspef", right_index=True)
            .groupby("ward_group")
            .speldur.agg(lambda x: sum(x + 1))  # convert los to bed days
        )
        self._beds_baseline.name = "baseline"

    def _load_theatres_data(self) -> None:
        """Load the Theatres data

        Loads the theatres data and stores in the private variable `self._theatres_data`
        """
        with open(f"{self._data_path}/theatres.json", "r", encoding="UTF-8") as tdf:
            self._theatres_data = json.load(tdf)
        self._theatres_data["four_hour_sessions"] = pd.Series(
            self._theatres_data["four_hour_sessions"], name="four_hour_sessions"
        )

    def _los_reduction(self, run_params: dict) -> pd.DataFrame:
        """Create a dictionary of the LOS reduction factors to use for a run

        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict
        :returns: a DataFrame containing the LOS reduction factors
        :rtype: pandas.DataFrame
        """
        params = self.params["inpatient_factors"]["los_reduction"]
        # convert the parameters dictionary to a dataframe: each item becomes a row (with the item
        # being the name of the row in the index), and then each sub-item becoming a column
        losr = pd.DataFrame.from_dict(params, orient="index")
        losr["losr_f"] = [
            run_params["inpatient_factors"]["los_reduction"][i] for i in losr.index
        ]
        return losr

    def _random_strategy(
        self, rng: np.random.Generator, strategy_type: str
    ) -> pd.DataFrame:
        """Select one strategy per record

        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param strategy_type: the type of strategy to update, e.g. "admission_avoidance",
            "los_reduction"
        :type strategy_type: str

        :returns: an updated DataFrame with a new column for the selected strategy
        :rtype: pandas.DataFrame
        """
        strategies = self._strategies[strategy_type]
        # sample from the strategies based on the sample_rate column, then select just the strategy
        # column
        strategies = strategies[
            rng.binomial(1, strategies["sample_rate"]).astype(bool)
        ].iloc[:, 0]
        # filter the strategies to only include those listed in the params file
        valid_strategies = list(
            self.params["inpatient_factors"][strategy_type].keys()
        ) + ["NULL"]
        strategies = strategies[strategies.isin(valid_strategies)]
        return (
            strategies
            # take all of the rows and randomly reshuffle them into a new order. We *do not* want to
            # use resampling here. make sure to use the same random state using rng.bit_generator
            .sample(frac=1, random_state=rng.bit_generator)
            # for each rn, select a single row, i.e. select 1 strategy per rn
            .groupby(level=0).head(1)
        )

    def _waiting_list_adjustment(self, data: pd.DataFrame) -> pd.Series:
        """Create a series of factors for waiting list adjustment.

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame

        :returns: a series of floats indicating how often we want to sample that row
        :rtype: pandas.Series
        """
        tretspef_n = data.groupby("tretspef").size()
        wla_param = pd.Series(self.params["waiting_list_adjustment"]["ip"])
        wla = (wla_param / tretspef_n).fillna(0) + 1
        # waiting list adjustment values
        # create a series of 1's the length of our data: make sure these are floats, not ints
        wlav = np.ones_like(data.index).astype(float)
        # find the rows which are elective wait list admissions
        i = list(data.admimeth == "11")
        # update the series for these rows with the waiting list adjustments for that specialty
        wlav[i] = wla[data[i].tretspef]
        # return the waiting list adjustment factor series
        return wlav

    @staticmethod
    def _non_demographic_adjustment(data, run_params: dict) -> pd.Series:
        """Create a series of factors for non-demographic adjustment.

        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: a series of floats indicating how often we want to sample that row)
        :rtype: pandas.Series
        """
        ndp = (
            pd.DataFrame.from_dict(
                run_params["non-demographic_adjustment"], orient="index"
            )
            .reset_index()
            .rename(columns={"index": "admigroup"})
            .melt(["admigroup"], var_name="age_group")
            .set_index(["age_group", "admigroup"])["value"]
        )

        return (
            data[["age_group", "admigroup"]]
            .join(ndp, on=["age_group", "admigroup"])["value"]
            .to_numpy()
        )

    @staticmethod
    def _admission_avoidance_step(
        admission_avoidance: pd.Series,
        run_params: dict,
    ) -> np.ndarray:
        """Run the admission avoidance strategies

        Runs the admission avoidance step and updates the `step_counts` dictionary

        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param admissions: the count of how many times each row has been sampled
        :type admissions: numpy.ndarray
        :param admission_avoidance: the selected admission avoidance strategies for this model run
        :type admission_avoidance: pandas.Series
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: the updated admissions array
        :rtype: numpy.ndarray
        """
        # choose admission avoidance factors
        ada = defaultdict(
            lambda: 1, run_params["inpatient_factors"]["admission_avoidance"]
        )
        return [ada[k] for k in admission_avoidance]

    @staticmethod
    def _losr_all(
        data: pd.DataFrame,
        losr: pd.DataFrame,
        rng: np.random.Generator,
        step_counts: dict,
    ) -> None:
        """Length of Stay Reduction: All

        Reduces all rows length of stay by sampling from a binomial distribution, using the current
        length of stay as the value for n, and the length of stay reduction factor for that strategy
        as the value for p. This will update the los to be a value between 0 and the original los.

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param losr: the Length of Stay rates table created from self._los_reduction()
        :type losr: pandas.DataFrame
        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param step_counts: a dictionary containing the changes to measures for this step
        :type step_counts: dict
        """
        i = losr.index[(losr.type == "all") & (losr.index.isin(data.index))]
        pre_los = data.loc[i, "speldur"]
        data.loc[i, "speldur"] = rng.binomial(
            data.loc[i, "speldur"], losr.loc[data.loc[i].index, "losr_f"]
        )
        change_los = (
            (data.loc[i, "speldur"] - pre_los).groupby(level=0).sum().astype(int)
        ).to_dict()

        for k in change_los.keys():
            step_counts[("los_reduction", k)] = {
                "admissions": 0,
                "beddays": change_los[k],
            }

    @staticmethod
    def _losr_bads(
        data: pd.DataFrame,
        losr: pd.DataFrame,
        rng: np.random.Generator,
        step_counts: dict,
    ) -> None:
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

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param losr: the Length of Stay rates table created from self._los_reduction()
        :type losr: pandas.DataFrame
        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param step_counts: a dictionary containing the changes to measures for this step
        :type step_counts: dict
        """
        i = losr.type == "bads"
        # create a copy of our data, and join to the losr data
        bads_df = data.merge(
            losr[i][["baseline_target_rate", "op_dc_split", "losr_f"]],
            left_index=True,
            right_index=True,
        )
        # convert the factor value to be the amount we need to adjust the non-target type to to make
        # the target rate equal too the factor value
        factor = (
            (bads_df["losr_f"] - bads_df["baseline_target_rate"])
            / (1 - bads_df["baseline_target_rate"])
        ).apply(inrange)
        # create three values that will sum to 1 - these will be the probabilties of:
        #   - staying where we are  [0.0, u0)
        #   - moving to daycase     [u0,  u1)
        #   - moving to outpatients (u1,  1.0]
        ur1 = bads_df["op_dc_split"] * factor
        ur2 = (1 - bads_df["op_dc_split"]) * factor
        ur0 = 1 - ur1 - ur2
        # we now create a random value for each row in [0, 1]
        rvr = rng.uniform(size=len(factor))
        # we can use this random value to update the patient class column appropriately
        bads_df.loc[
            (rvr >= ur0) & (rvr < ur0 + ur1), "classpat"
        ] = "2"  # row becomes daycase
        bads_df.loc[(rvr >= ur0 + ur1), "classpat"] = "-1"  # row becomes outpatients
        # we now need to apply these changes to the actual data
        i = losr.index[i]
        # make sure we only keep values in losr that exist in data
        i = i[i.isin(data.index)]
        data.loc[i, "classpat"] = bads_df["classpat"]
        # set the speldur to 0 if we aren't inpatients
        data.loc[i, "speldur"] *= data.loc[i, "classpat"] == "1"

        step_counts_admissions = (
            ((data.loc[i, "classpat"] == "-1").groupby(level=0).sum() * -1)
            .astype(int)
            .to_dict()
        )
        step_counts_beddays = (
            (data.loc[i, "speldur"] - bads_df["speldur"])
            .groupby(level=0)
            .sum()
            .astype(int)
            .to_dict()
        )

        for k in step_counts_admissions.keys():
            step_counts[("los_reduction", k)] = {
                "admissions": step_counts_admissions[k],
                "beddays": step_counts_beddays[k],
            }

    @staticmethod
    def _losr_to_zero(
        data: pd.DataFrame,
        losr: pd.DataFrame,
        rng: np.random.Generator,
        losr_type: str,
        step_counts: dict,
    ) -> None:
        """
        Length of Stay Reduction: To Zero Day LoS

        Updates the length of stay to 0 for a given percentage of rows.

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param losr: the Length of Stay rates table created from self._los_reduction()
        :type losr: pandas.DataFrame
        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param step_counts: a dictionary containing the changes to measures for this step
        :type step_counts: dict
        """
        i = losr.index[(losr.type == losr_type) & (losr.index.isin(data.index))]
        pre_los = data.loc[i, "speldur"]
        nrow = len(data.loc[i, "speldur"])
        data.loc[i, "speldur"] *= (
            rng.uniform(size=nrow) >= losr.loc[data.loc[i].index, "losr_f"]
        )
        change_los = (
            (data.loc[i, "speldur"] - pre_los).groupby(level=0).sum().astype(int)
        ).to_dict()

        for k in change_los.keys():
            step_counts[("los_reduction", k)] = {
                "admissions": 0,
                "beddays": change_los[k],
            }

    def _run(
        self,
        rng: np.random.Generator,
        data: pd.DataFrame,
        run_params: dict,
        demo_f: pd.Series,
        hsa_f: pd.Series,
    ) -> tuple[dict, pd.DataFrame]:
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
        # select strategies
        admission_avoidance = self._random_strategy(rng, "admission_avoidance")
        admission_avoidance = admission_avoidance[data["rn"]]
        los_reduction = self._random_strategy(rng, "los_reduction")
        # choose length of stay reduction factors
        losr = self._los_reduction(run_params)

        step_counts = {
            ("baseline", "-"): {
                "admissions": len(data.index),
                "beddays": int(sum(data["speldur"] + 1)),
            }
        }

        # create an array that keeps track of the number of times we have sampled each row
        admissions = (
            InpatientsAdmissionsCounter(data, rng)
            .poisson_step(hsa_f, "health_status_adjustment")
            .poisson_step(demo_f[data["rn"]], "population_factors")
            .poisson_step(
                self._waiting_list_adjustment(data), "waiting_list_adjustment"
            )
            .poisson_step(
                self._non_demographic_adjustment(data, run_params),
                "non-demographic_adjustment",
            )
            .binomial_step(
                self._admission_avoidance_step(admission_avoidance, run_params),
                "admission_avoidance",
                admission_avoidance.tolist(),
            )
        )
        # done with resampling rows, get the resampled data and the step counts
        data = admissions.get_data()
        step_counts = admissions.step_counts
        # LoS Reduction ----------------------------------------------------------------------------
        # set the index for easier querying
        data.set_index(los_reduction[data["rn"]], inplace=True)
        # run each of the length of stay reduction strategies
        self._losr_all(data, losr, rng, step_counts)
        self._losr_to_zero(data, losr, rng, "aec", step_counts)
        self._losr_to_zero(data, losr, rng, "preop", step_counts)
        self._losr_bads(data, losr, rng, step_counts)
        # return the data (select just the columns we have updated in modelling)
        change_factors = pd.melt(
            pd.DataFrame.from_dict(step_counts, "index")
            .rename_axis(["change_factor", "strategy"])
            .reset_index(),
            ["change_factor", "strategy"],
            ["admissions", "beddays"],
            "measure",
        )
        change_factors["value"] = change_factors["value"].astype(int)
        return (
            change_factors,
            data.drop(["hsagrp"], axis="columns").reset_index(drop=True),
        )

    def _bed_occupancy(
        self, ip_rows: pd.DataFrame, bed_occupancy_params: dict, model_run: int
    ) -> dict[namedtuple, int]:
        """Calculate bed occupancy

        :param ip_rows: the Inpatient rows from model results
        :type ip_rows: pandas.DataFrame
        :param bed_occupancy_params: the bed occupancy parameters from run_params
        :type bed_occupancy_params: dict
        :param model_run: the current model run
        :type model_run: int

        :returns: a dictionary of a named tuple to an integer. The named tuple contains the pod,
            measure, and the ward group. The value is the number of beds available
        :rtype: dict
        """
        # create the namedtuple type
        result = namedtuple("results", ["pod", "measure", "ward_group"])
        if model_run == -1:
            return {
                result("ip", "day+night", k): v
                for k, v in self._kh03_data["available"].iteritems()
            }

        target_bed_occupancy_rates = pd.Series(bed_occupancy_params["day+night"])
        # get the model run data
        dn_admissions = (
            ip_rows[
                (ip_rows["measure"] == "beddays")
                & (
                    ip_rows["pod"].isin(
                        ["ip_non-elective_admission", "ip_elective_admission"]
                    )
                )
            ]
            .merge(self._ga_ward_groups, left_on="mainspef", right_index=True)
            .groupby("ward_group")
            .value.sum()
        )
        # return the results
        return {
            result("ip", "day+night", k): v
            for k, v in (
                (
                    (dn_admissions * self._kh03_data["occupied"])
                    / (self._beds_baseline * target_bed_occupancy_rates)
                )
                .dropna()
                .to_dict()
            ).items()
        }

    def _theatres_available(
        self, model_results: pd.DataFrame, theatres_params: dict, model_run: int
    ) -> dict[namedtuple, int]:
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
        # create the namedtuple types
        result_u = namedtuple("results", ["pod", "measure", "tretspef"])
        result_a = namedtuple("results", ["pod", "measure"])

        fhs = self._theatres_data["four_hour_sessions"]
        theatres = self._theatres_data["theatres"]

        if model_run == -1:
            return {
                **{
                    result_u("ip_theatres", "four_hour_sessions", k): v
                    for k, v in fhs.iteritems()
                },
                result_a("ip_theatres", "theatres"): theatres,
            }

        change_availability = theatres_params["change_availability"]
        change_utilisation = pd.Series(
            theatres_params["change_utilisation"], name="change_utilisation"
        )

        activity = pd.concat(
            {
                k: (
                    v[v.has_procedure == 1]
                    .assign(is_elective=lambda x: x.admigroup == "elective", n=1)
                    .groupby("tretspef", as_index=False)[["is_elective", "n"]]
                    .sum()
                    # keep just the specialties in the fhs object
                    .assign(
                        tretspef=lambda x: np.where(
                            x.tretspef.isin(fhs.index.to_list()), x.tretspef, "Other"
                        )
                    )
                    .groupby("tretspef")
                    .sum()
                )
                for k, v in [
                    ("baseline", self.data),
                    ("future", model_results),
                ]
            }
        )

        activity["four_hour_sessions"] = pd.concat(
            {
                "baseline": fhs,
                "future": activity.loc["future", "is_elective"]
                / activity.loc["baseline", "is_elective"]
                * fhs
                / change_utilisation,
            }
        )

        fhs_with_non_el = (
            (activity["n"] / activity["is_elective"] * activity["four_hour_sessions"])
            .groupby(level=0)
            .sum()
        )
        fhs_with_non_el_change = fhs_with_non_el["future"] / fhs_with_non_el["baseline"]

        new_theatres = fhs_with_non_el_change * theatres / change_availability

        return {
            **{
                result_u("ip_theatres", "four_hour_sessions", k): v
                for k, v in activity.loc["future", "four_hour_sessions"].iteritems()
            },
            result_a("ip_theatres", "theatres"): new_theatres,
        }

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
        # get the run params: use principal run for baseline
        run_params = self._get_run_params(max(0, model_run))

        # row's with a classpat of -1 are outpatients and need to be handled separately
        ip_op_row_ix = model_results["classpat"] == "-1"
        ip_rows = model_results[~ip_op_row_ix].copy()
        op_rows = model_results[ip_op_row_ix].copy()
        # run the theatres utilisation on the model results before we alter ip_rows
        theatres_available = self._theatres_available(
            ip_rows, run_params["theatres"], model_run
        )
        # handle OP rows
        op_rows["pod"] = "op_procedure"
        op_rows["measure"] = "attendances"
        op_rows["value"] = 1
        # handle IP rows
        # create an admission group column
        ip_rows["admission_group"] = "ip_non-elective"
        ip_rows.loc[
            ip_rows["admimeth"].str.startswith("1"), "admission_group"
        ] = "ip_elective"
        # quick dq fix: convert any "non-elective" daycases to "elective"
        ip_rows.loc[
            ip_rows["classpat"].isin(["2", "3"]), "admission_group"
        ] = "ip_elective"
        # create a "pod" column, starting with the admission group
        ip_rows["pod"] = ip_rows["admission_group"]
        ip_rows.loc[ip_rows["classpat"].isin(["1", "4"]), "pod"] += "_admission"
        ip_rows.loc[ip_rows["classpat"].isin(["2", "3"]), "pod"] += "_daycase"
        ip_rows.loc[ip_rows["classpat"] == "5", "pod"] += "_birth-episode"

        ip_rows["beddays"] = ip_rows["speldur"] + 1
        ip_rows = (
            ip_rows.assign(admissions=1)
            .reset_index()
            .melt(
                ["rn", "age_group", "sex", "tretspef", "mainspef", "pod"],
                ["admissions", "beddays"],
                "measure",
            )
        )

        cols = list(set(ip_rows.columns).intersection(set(op_rows.columns)))
        model_results = (
            pd.concat([op_rows[cols], ip_rows[cols]])
            # summarise the results to make the create_agg steps quicker
            .groupby(
                # note: any columns used in the calls to _create_agg, including pod and measure
                # must be included below
                ["pod", "measure", "sex", "age_group", "tretspef"],
                as_index=False,
            )[["value"]].sum()
        )

        agg = partial(self._create_agg, model_results)
        return {
            **agg(),
            **agg(["sex", "age_group"]),
            **agg(["sex", "tretspef"]),
            "bed_occupancy": self._bed_occupancy(
                ip_rows, run_params["bed_occupancy"], model_run
            ),
            "theatres_available": theatres_available,
        }

    def save_results(
        self, model_results: pd.DataFrame, path_fn: Callable[[str], str]
    ) -> None:
        """Save the results of running the model

        :param model_results: a DataFrame containing the results of a model iteration
        :type model_results: pandas.DataFrame
        :param path_fn: a function that takes the activity type and generates a path where to save the file
        :type path_fn: Callable[[str], str]
        """
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
