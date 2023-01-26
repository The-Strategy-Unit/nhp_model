"""
Inpatients Module

Implements the inpatients model.
"""
import json
from collections import defaultdict, namedtuple
from datetime import timedelta
from functools import partial, reduce
from typing import Callable

import numpy as np
import pandas as pd

from model.helpers import inrange
from model.model import Model


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
                "admigroup",
                "classpat",
                "mainspef",
                "tretspef",
                "hsagrp",
                "has_procedure",
                "is_main_icb",
                "admidate",
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
        # oversample
        self._union_bedday_rows()

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
            .groupby(["quarter", "ward_group"])[["available", "occupied"]]
            .agg(lambda x: sum(np.round(x).astype(int)))
        )
        # get the baseline data
        self._beds_baseline = self._bedday_summary(self.data, self.params["start_year"])

    def _union_bedday_rows(self) -> None:
        """Oversample the rows that are admitted in the prior financial year to simulate those
        who were discharged in the next financial year.
        """
        start_year = self.params["start_year"]
        pre_rows = self.data[
            pd.to_datetime(self.data.admidate) < pd.to_datetime(f"{start_year}-04-01")
        ].copy()
        pre_rows.admidate += timedelta(days=365)
        # add column to indicate whether this is a row used for bedday calculations
        pre_rows["bedday_rows"] = True
        self.data["bedday_rows"] = False
        # append these rows of data to the actual data
        self.data = pd.concat([self.data, pre_rows]).reset_index(drop=True)

    def _load_theatres_data(self) -> None:
        """Load the Theatres data

        Loads the theatres data and stores in the private variable `self._theatres_data`
        """
        with open(f"{self._data_path}/theatres.json", "r", encoding="UTF-8") as tdf:
            self._theatres_data = json.load(tdf)

        fhs_baseline = pd.Series(
            self._theatres_data["four_hour_sessions"], name="four_hour_sessions"
        ).rename(index={"Other": "Other (Surgical)"})

        baseline_params = pd.Series(
            {
                k: v["baseline"]
                for k, v in self.params["theatres"]["change_utilisation"].items()
            }
        )

        self._theatres_data["four_hour_sessions"] = fhs_baseline
        self._theatres_baseline = self.data.groupby("tretspef")["has_procedure"].sum()
        self._theatre_spells_baseline = (fhs_baseline / baseline_params).sum()

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

    @staticmethod
    def _expat_adjustment(data: pd.DataFrame, run_params: dict) -> pd.Series:
        expat_params = run_params["expat"]["ip"]
        join_cols = ["admigroup", "tretspef"]
        return data.merge(
            pd.DataFrame(
                [
                    {"admigroup": k1, "tretspef": k2, "value": v2}
                    for k1, v1 in expat_params.items()
                    for k2, v2 in v1.items()
                ]
            ).set_index(join_cols),
            how="left",
            left_on=join_cols,
            right_index=True,
        )["value"].fillna(1)

    @staticmethod
    def _repat_adjustment(data: pd.DataFrame, run_params: dict) -> pd.Series:
        repat_local_params = run_params["repat_local"]["ip"]
        repat_nonlocal_params = run_params["repat_nonlocal"]["ip"]
        join_cols = ["admigroup", "tretspef", "is_main_icb"]
        return data.merge(
            pd.DataFrame(
                [
                    {"admigroup": k1, "tretspef": k2, "is_main_icb": icb, "value": v2}
                    for (k0, icb) in [
                        (repat_local_params, True),
                        (repat_nonlocal_params, False),
                    ]
                    for k1, v1 in k0.items()
                    for k2, v2 in v1.items()
                ]
            ).set_index(join_cols),
            how="left",
            left_on=join_cols,
            right_index=True,
        )["value"].fillna(1)

    @staticmethod
    def _baseline_adjustment(data: pd.DataFrame, run_params: dict) -> pd.Series:
        """Create a series of factors for baseline adjustment.

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: a series of floats indicating how often we want to sample that row
        :rtype: pandas.Series
        """
        params = pd.concat(
            {
                k: pd.Series(v, name="baseline_adjustment")
                for k, v in run_params["baseline_adjustment"]["ip"].items()
            }
        )
        return data.merge(
            params, how="left", left_on=["admigroup", "tretspef"], right_index=True
        )["baseline_adjustment"].fillna(1)

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
    def _admission_avoidance(
        admission_avoidance: pd.Series,
        run_params: dict,
    ) -> np.ndarray:
        """Calculate the admission avoidance change factors

        :param admission_avoidance: the selected admission avoidance strategies for this model run
        :type admission_avoidance: pandas.Series
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: the admission avoidance factors
        :rtype: numpy.ndarray
        """
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
        data.loc[i, "classpat"] = bads_df["classpat"].tolist()
        # set the speldur to 0 if we aren't inpatients
        data.loc[i, "speldur"] *= data.loc[i, "classpat"] == "1"

        step_counts_admissions = (
            ((data.loc[i, "classpat"] == "-1").groupby(level=0).sum() * -1)
            .astype(int)
            .to_dict()
        )
        step_counts_beddays = (
            (
                data.loc[i, "speldur"].groupby(level=0).sum()
                - bads_df["speldur"].groupby(level=0).sum()
            )
            .astype(int)
            .to_dict()
        )

        for k in step_counts_admissions.keys():
            step_counts[("los_reduction", k)] = {
                "admissions": step_counts_admissions[k],
                "beddays": step_counts_beddays[k],
            }

    @staticmethod
    def _losr_aec(
        data: pd.DataFrame,
        losr: pd.DataFrame,
        rng: np.random.Generator,
        step_counts: dict,
    ) -> None:
        """
        Length of Stay Reduction: AEC reduction

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
        i = losr.index[(losr.type == "aec") & (losr.index.isin(data.index))]
        pre_los = data.loc[i, "speldur"]
        data.loc[i, "speldur"] *= rng.binomial(1, losr.loc[data.loc[i].index, "losr_f"])
        change_los = (
            (data.loc[i, "speldur"] - pre_los).groupby(level=0).sum().astype(int)
        ).to_dict()

        for k in change_los.keys():
            step_counts[("los_reduction", k)] = {
                "admissions": 0,
                "beddays": change_los[k],
            }

    @staticmethod
    def _losr_preop(
        data: pd.DataFrame,
        losr: pd.DataFrame,
        rng: np.random.Generator,
        step_counts: dict,
    ) -> None:
        """
        Length of Stay Reduction: Pre-op reduction

        Updates the length of stay to by removing 1 or 2 days for a given percentage of rows

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param losr: the Length of Stay rates table created from self._los_reduction()
        :type losr: pandas.DataFrame
        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param step_counts: a dictionary containing the changes to measures for this step
        :type step_counts: dict
        """
        i = losr.index[(losr.type == "pre-op") & (losr.index.isin(data.index))]
        pre_los = data.loc[i, "speldur"]
        data.loc[i, "speldur"] -= (
            rng.binomial(1, 1 - losr.loc[data.loc[i].index, "losr_f"])
            * losr.loc[data.loc[i].index, "pre-op_days"]
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
    def _step_counts(
        admissions: pd.Series,
        beddays_before: pd.Series,
        beddays_after: int,
        admission_avoidance: pd.Series,
        factors: dict,
    ) -> dict:
        # TODO: add docstring and datatypes
        admissions_before = len(admissions)
        admissions_after = sum(admissions)

        # convert the beddays before modelling to a complex number:
        #   * the "real" part is the beddays
        #   * the "imaginary" part is the number of admissions
        # this makes it easy to keep track of both admissions and beddays count
        n = (beddays_before + 1j).to_numpy()

        # get the next value of n
        nt = n * admission_avoidance
        cf = {
            ("admission_avoidance", k): {"admissions": v.imag, "beddays": v.real}
            for k, v in (n - nt).groupby(level=0).sum().to_dict().items()
            if v != 0
        }
        # replace n, and then work out the sum of n
        n = nt.to_numpy()
        ns = n.sum()

        # then apply all other steps
        for k, v in factors.items():
            nt = n * v
            nts = nt.sum()
            x = nts - ns
            cf[(k, "-")] = {"admissions": x.imag, "beddays": x.real}
            n, ns = nt, nts

        # work out the "slack" between the model results and the estimated change factors
        admissions_slack = 1 + (admissions_after - ns.imag) / (
            ns.imag - admissions_before
        )
        beddays_slack = 1 + (beddays_after - ns.real) / (ns.real - beddays_before.sum())

        return {
            ("baseline", "-"): {
                "admissions": admissions_before,
                "beddays": beddays_before.sum(),
            },
            **{
                k: {
                    "admissions": v["admissions"] * admissions_slack,
                    "beddays": v["beddays"] * beddays_slack,
                }
                for k, v in cf.items()
            },
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
        admission_avoidance = self._random_strategy(rng, "admission_avoidance")[
            data["rn"]
        ]
        los_reduction = self._random_strategy(rng, "los_reduction")
        # choose length of stay reduction factors
        losr = self._los_reduction(run_params)

        # Row Resampling ---------------------------------------------------------------------------
        aa = self._admission_avoidance(admission_avoidance, run_params)
        factors = {
            "health_status_adjustment": hsa_f,
            "population_factors": demo_f[data["rn"]].to_numpy(),
            "expatriation": self._expat_adjustment(data, run_params).to_numpy(),
            "repatriation": self._repat_adjustment(data, run_params).to_numpy(),
            "baseline_adjustment": self._baseline_adjustment(
                data, run_params
            ).to_numpy(),
            "waiting_list_adjustment": self._waiting_list_adjustment(data),
            "non-demographic_adjustment": self._non_demographic_adjustment(
                data, run_params
            ),
        }

        reduced_factor = np.prod(list(factors.values()), axis=0) * aa

        # create an array that keeps track of the number of times we have sampled each row
        admissions = rng.poisson(reduced_factor)

        # beddays before resamping rows
        beddays_before = data["speldur"] + 1

        # resample rows
        data = data.loc[data.index.repeat(admissions)].reset_index(drop=True)

        r = run_params["inpatient_factors"]["admission_avoidance"]

        # calculate beddays after resampling rows
        beddays_after = data["speldur"] + 1
        # calculate step counts
        step_counts = self._step_counts(
            admissions,
            beddays_before,
            beddays_after.sum(),
            # admission_avoidance contains the strategy assigned to each row
            # convert that to the factor for the given strategy
            # then set the index to be the name of the strategy
            admission_avoidance.apply(lambda v: r.get(v, 1.0)).set_axis(
                admission_avoidance
            ),
            factors,
        )

        # LoS Reduction ----------------------------------------------------------------------------
        # set the index for easier querying
        data.set_index(los_reduction.loc[data["rn"]], inplace=True)
        # run each of the length of stay reduction strategies
        self._losr_all(data, losr, rng, step_counts)
        self._losr_aec(data, losr, rng, step_counts)
        self._losr_preop(data, losr, rng, step_counts)
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
        return (
            change_factors,
            data.drop(["hsagrp"], axis="columns").reset_index(drop=True),
        )

    def _bedday_summary(self, data, year):
        """Summarise data to count how many beddays per quarter

        Calculated midnight bed occupancy per day, summarises to maximum in quarter.

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
            .assign(day_n=lambda x: x.speldur.apply(np.arange))
            .explode("day_n")
            .groupby(["admidate", "day_n", "mainspef"], as_index=False)
            .size()
            .assign(
                date=lambda x: pd.to_datetime(x.admidate)
                + pd.to_timedelta(x.day_n, unit="d")
            )
            .merge(self._ward_groups, left_on="mainspef", right_index=True)
            .groupby(["date", "ward_type", "ward_group"], as_index=False)
            .agg({"size": sum})
            .assign(quarter=lambda x: x.date.dt.to_period("Q-MAR"))
            .query(f"quarter.dt.qyear == {year + 1}")
            .groupby(["quarter", "ward_type", "ward_group"], as_index=False)
            .agg({"size": max})
            .assign(quarter=lambda x: x.quarter.astype(str).str[4:].str.lower())
        )

    def _bed_occupancy(
        self, data: pd.DataFrame, bed_occupancy_params: dict, model_run: int
    ) -> dict[namedtuple, int]:
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
        # create the namedtuple type
        result = namedtuple("results", ["pod", "measure", "quarter", "ward_group"])
        if model_run == -1:
            # todo: load kh03 data by quarter
            return {
                result("ip", "day+night", q, k): np.round(v).astype(int)
                for (q, k), v in self._kh03_data["available"].iteritems()
            }

        target_bed_occupancy_rates = pd.Series(bed_occupancy_params["day+night"])
        target_bed_occupancy_rates.name = "target_occupancy_rate"

        beddays_model = (
            self._bedday_summary(data, self.params["start_year"])
            .rename(columns={"size": "model"})
            .merge(
                self._kh03_data["occupied"],
                left_on=["quarter", "ward_group"],
                right_index=True,
                how="outer",
            )
            .fillna(value={"occupied": 0})
            .set_index(["quarter", "ward_group"])
        )
        beddays_baseline = (
            self._beds_baseline.rename(columns={"size": "baseline"})
            .merge(target_bed_occupancy_rates, left_on="ward_group", right_index=True)
            .set_index(["quarter", "ward_group"])
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
            result("ip", "day+night", p, s): np.round(v).astype(int)
            for (p, s), v in beddays_results.items()
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

        fhs_baseline = self._theatres_data["four_hour_sessions"]
        theatres = self._theatres_data["theatres"]

        if model_run == -1:
            return {
                **{
                    result_u("ip_theatres", "four_hour_sessions", k): v
                    for k, v in fhs_baseline.iteritems()
                },
                result_a("ip_theatres", "theatres"): theatres,
            }
        change_availability = theatres_params["change_availability"]
        change_utilisation = pd.Series(
            theatres_params["change_utilisation"], name="change_utilisation"
        )

        future = (
            model_results[model_results["measure"] == "procedures"]
            .groupby("tretspef")["value"]
            .sum()
        )

        baseline = self._procedures_baseline
        theatre_spells_baseline = self._theatre_spells_baseline

        fhs_future = (future / baseline * fhs_baseline).dropna()
        theatre_spells_future = fhs_future / change_utilisation

        theatre_spells_change = theatre_spells_future.sum() / theatre_spells_baseline

        new_theatres = theatres * theatre_spells_change / change_availability
        return {
            **{
                result_u("ip_theatres", "four_hour_sessions", k): v
                for k, v in fhs_future.iteritems()
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

        bed_occupancy = self._bed_occupancy(
            model_results,
            run_params["bed_occupancy"],
            model_run,
        )

        model_results = (
            model_results[~model_results["bedday_rows"]]
            .groupby(
                [
                    "age_group",
                    "sex",
                    "admimeth",
                    "admigroup",
                    "classpat",
                    "mainspef",
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

        model_results.loc[
            model_results["admigroup"] == "maternity", "admigroup"
        ] = "non-elective"
        model_results["pod"] = "ip_" + model_results["admigroup"] + "_admission"
        # quick dq fix: convert any "non-elective" daycases to "elective"
        model_results.loc[
            model_results["classpat"].isin(["2", "3"]), "pod"
        ] = "ip_elective_daycase"
        model_results.loc[
            model_results["classpat"] == "5", "pod"
        ] = "ip_non-elective_birth-episode"

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
        return {
            **agg(),
            **agg(["sex", "age_group"]),
            **agg(["sex", "tretspef"]),
            "bed_occupancy": bed_occupancy,
            "theatres_available": self._theatres_available(
                model_results.loc[~op_rows],
                run_params["theatres"],
                model_run,
            ),
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
