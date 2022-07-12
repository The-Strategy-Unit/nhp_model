"""
Inpatients Module

Implements the inpatients model.
"""
import os
from collections import defaultdict, namedtuple
from functools import partial

import numpy as np
import pandas as pd

from model.helpers import inrange
from model.model import Model


class InpatientsModel(Model):
    """
    Inpatients Model

    * params: the parameters to run the model with
    * data_path: the path to where the data files live

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
            ],
        )
        # load the strategies, store each strategy file as a separate entry in a dictionary
        self._strategies = {
            x: self._load_parquet(f"ip_{x}_strategies").set_index(["rn"])
            for x in ["admission_avoidance", "los_reduction"]
        }

    def _los_reduction(self, run_params: dict) -> pd.DataFrame:
        """
        Create a dictionary of the LOS reduction factors to use for a run

        * run_params: the parameters to use for this model run (see `Model._get_run_params()`)

        returns: a DataFrame containing the LOS reduction factors
        """
        params = self.params["strategy_params"]["los_reduction"]
        # convert the parameters dictionary to a dataframe: each item becomes a row (with the item
        # being the name of the row in the index), and then each sub-item becoming a column
        losr = pd.DataFrame.from_dict(params, orient="index")
        losr["losr_f"] = [
            run_params["strategy_params"]["los_reduction"][i] for i in losr.index
        ]
        return losr

    def _random_strategy(
        self, rng: np.random.Generator, strategy_type: str
    ) -> pd.DataFrame:
        """
        Select one strategy per record

        * rng: an np.random.Generator, created for each model iteration
        * strategy_type: a string of which type of strategy to update, e.g. "admission_avoidance",
          "los_reduction"

        returns: an updated DataFrame with a new column for the selected strategy
        """
        strategies = self._strategies[strategy_type]
        # sample from the strategies based on the sample_rate column, then select just the strategy
        # column
        strategies = strategies[
            rng.binomial(1, strategies["sample_rate"]).astype(bool)
        ].iloc[:, 0]
        # filter the strategies to only include those listed in the params file
        valid_strategies = list(
            self.params["strategy_params"][strategy_type].keys()
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
        """
        Create a series of factors for waiting list adjustment.

        * data: the DataFrame that we are updating

        returns: a series of floats indicating how often we want to sample that row

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline
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
        """
        Create a series of factors for non-demographic adjustment.

        * run_params: the parameters to use for this model run (see `Model._get_run_params()`

        returns: a series of floats indicating how often we want to sample that row)
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
        admigroup = np.where(
            data["admimeth"].str.startswith("1"),
            "elective",
            np.where(data["admimeth"].str.startswith("3"), "maternity", "non-elective"),
        )

        return (
            data[["age_group"]]
            .assign(admigroup=admigroup)
            .join(ndp, on=["age_group", "admigroup"])["value"]
            .to_numpy()
        )

    @staticmethod
    def _admission_avoidance_step(
        rng: np.random.Generator,
        data: pd.DataFrame,
        admission_avoidance: pd.Series,
        run_params: dict,
        step_counts: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the admission avoidance strategies

        * rng: an np.random.Generator, created for each model iteration
        * data: the DataFrame that we are updating
        * admission_avoidance: the selected admission avoidance strategies for this model run
        * run_params: the parameters to use for this model run (see `Model._get_run_params()`)

        returns: a tuple containing the updated data and the change factors for this step
        """
        # choose admission avoidance factors
        ada = defaultdict(
            lambda: 1, run_params["strategy_params"]["admission_avoidance"]
        )
        # work out the number of rows/sum of los before admission avoidance
        data_aa = data.merge(
            admission_avoidance, left_on="rn", right_index=True
        ).groupby("admission_avoidance_strategy")
        sc_n_aa = data_aa["rn"].agg(len)
        # add rows to convert los to beddays
        sc_b_aa = data_aa["speldur"].agg(sum) + sc_n_aa
        # then, work out the admission avoidance factors for each row
        aaf = [ada[k] for k in admission_avoidance[data["rn"]]]
        # decide whether to select this row or not
        select_row = rng.binomial(n=1, p=aaf).astype(bool)
        # select each row as many times as it was sampled
        data = data[select_row].reset_index(drop=True)
        # update our number of rows and update step_counts
        data_aa = data.merge(
            admission_avoidance, left_on="rn", right_index=True
        ).groupby("admission_avoidance_strategy")
        # handle the case of a strategy eliminating all rows
        sc_n_aa_post = data_aa["rn"].agg(len)
        # as above, convert los to beddays
        sc_b_aa_post = data_aa["speldur"].agg(sum) + sc_n_aa_post
        # convert to default dict's: if a strategy eliminates all rows the final step will fail
        # returning NaN values for these strategies
        sc_n_aa_post = defaultdict(lambda: 0, sc_n_aa_post.to_dict())
        sc_b_aa_post = defaultdict(lambda: 0, sc_b_aa_post.to_dict())
        for k in sc_n_aa.index:
            step_counts[("admission_avoidance", k)] = {
                "admissions": sc_n_aa_post[k] - sc_n_aa[k],
                "beddays": sc_b_aa_post[k] - sc_b_aa[k],
            }
        return data

    @staticmethod
    def _losr_all(
        data: pd.DataFrame,
        losr: pd.DataFrame,
        rng: np.random.Generator,
        step_counts: dict,
    ) -> None:
        """
        Length of Stay Reduction: All

        * data: the DataFrame that we are updating
        * losr: the Length of Stay rates table created from self._los_reduction()
        * rng: an np.random.Generator, created for each model iteration
        * step_counts: a dictionary containing the changes to measures for this step

        Reduces all rows length of stay by sampling from a binomial distribution, using the current
        length of stay as the value for n, and the length of stay reduction factor for that strategy
        as the value for p. This will update the los to be a value between 0 and the original los.
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

        * data: the DataFrame that we are updating
        * losr: the Length of Stay rates table created from self._los_reduction()
        * rng: an np.random.Generator, created for each model iteration
        * step_counts: a dictionary containing the changes to measures for this step

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

        * data: the DataFrame that we are updating
        * losr: the Length of Stay rates table created from self._los_reduction()
        * rng: an np.random.Generator, created for each model iteration
        * type: the type of row we are updating
        * step_counts: a dictionary containing the changes to measures for this step

        Updates the length of stay to 0 for a given percentage of rows.
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

    @staticmethod
    def _run_poisson_step(rng, data, name, factor, step_counts):
        select_row_n_times = rng.poisson(factor)
        sc_n = len(data.index)
        sc_b = int(sum(data["speldur"] + 1))
        # perform the step
        data = data.loc[data.index.repeat(select_row_n_times)].reset_index(drop=True)
        # update the step count values
        sc_np = int(sum(select_row_n_times))
        sc_bp = int(sum(data["speldur"] + 1))
        step_counts[(name, "-")] = {"admissions": sc_np - sc_n, "beddays": sc_bp - sc_b}
        return data

    def _run(
        self,
        rng: np.random.Generator,
        data: pd.DataFrame,
        run_params: dict,
        aav_f: pd.Series,
        hsa_f: pd.Series,
    ) -> tuple[pd.DataFrame, dict]:
        """Run the model

        * rng: an np.random.Generator, created for each model iteration
        * data: the DataFrame that we are updating
        * run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        * aav_f: the demographic adjustment factors
        * hsa_f: the health status adjustment factors

        returns: a tuple containing the change factors DataFrame and the mode results DataFrame
        """
        # select strategies
        admission_avoidance = self._random_strategy(rng, "admission_avoidance")
        los_reduction = self._random_strategy(rng, "los_reduction")
        # choose length of stay reduction factors
        losr = self._los_reduction(run_params)

        step_counts = {
            ("baseline", "-"): {
                "admissions": len(data.index),
                "beddays": int(sum(data["speldur"] + 1)),
            }
        }

        # first, run hsa as we have the factor already created
        data = self._run_poisson_step(
            rng, data, "health_status_adjustment", hsa_f, step_counts
        )
        # then, demographic modelling
        data = self._run_poisson_step(
            rng, data, "population_factors", aav_f[data["rn"]], step_counts
        )
        # waiting list adjustments
        data = self._run_poisson_step(
            rng,
            data,
            "waiting_list_adjustment",
            self._waiting_list_adjustment(data),
            step_counts,
        )
        # non-demographic adjustment
        data = self._run_poisson_step(
            rng,
            data,
            "non-demographic_adjustment",
            self._non_demographic_adjustment(data, run_params),
            step_counts,
        )
        # Admission Avoidance ----------------------------------------------------------------------
        data = self._admission_avoidance_step(
            rng, data, admission_avoidance, run_params, step_counts
        )
        # done with row count adjustment
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

    def _bed_occupancy(self, ip_rows: pd.DataFrame, bed_occupancy_params: dict):
        # extract params
        ga_ward_groups = pd.Series(
            self.params["bed_occupancy"]["specialty_mapping"]["General and Acute"],
            name="ward_group",
        )
        bed_occupancy_rates = pd.Series(bed_occupancy_params["day+night"])
        # get the baseline data
        baseline = (
            self.data[self.data.classpat.isin(["1", "4"])]
            .merge(ga_ward_groups, left_on="mainspef", right_index=True)
            .groupby("ward_group")
            .speldur.sum()
        )
        baseline.name = "baseline"
        # get the model run data
        dn_admissions = (
            ip_rows[
                (ip_rows["measure"] == "beddays")
                & (ip_rows["pod"].str.endswith("admission"))
            ]
            .merge(ga_ward_groups, left_on="mainspef", right_index=True)
            .groupby("ward_group")
            .value.sum()
        )
        # load the kh03 data
        kh03_data = (
            pd.read_csv(
                f"{self._data_path}/kh03.csv",
                dtype={
                    "specialty_code": np.character,
                    "specialty_group": np.character,
                    "available": np.float64,
                    "occupied": np.float64,
                },
            )
            .merge(ga_ward_groups, left_on="specialty_code", right_index=True)
            .groupby(["ward_group"])
            .occupied.sum()
        )
        # create the namedtuple type
        result = namedtuple("results", ["pod", "measure", "ward_group"])
        # return the results
        return {
            result("ip", "day+night", k): v
            for k, v in (
                ((dn_admissions * kh03_data) / (baseline * bed_occupancy_rates))
                .dropna()
                .to_dict()
            ).items()
        }

    def aggregate(self, model_results: pd.DataFrame, model_run: int) -> dict:
        """
        Aggregate the model results

        * model_results: a DataFrame containing the results of a model iteration
        * model_run: the current model run

        returns: a dictionary containing the different aggregations of this data

        Can also be used to aggregate the baseline data by passing in the raw data
        """
        # get the run params: use principal run for baseline
        run_params = self._get_run_params(max(0, model_run))
        # row's with a classpat of -1 are outpatients and need to be handled separately
        ip_op_row_ix = model_results["classpat"] == "-1"
        ip_rows = model_results[~ip_op_row_ix].copy()
        op_rows = model_results[ip_op_row_ix].copy()
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
        model_results = pd.concat([op_rows[cols], ip_rows[cols]])

        agg = partial(self._create_agg, model_results)
        return {
            **agg(),
            **agg(["sex", "age_group"]),
            **agg(["sex", "tretspef"]),
            "bed_occupancy": self._bed_occupancy(ip_rows, run_params["bed_occupancy"]),
        }

    def save_results(self, results, path_fn):
        """Save the results of running the model"""
        ip_op_row_ix = results["classpat"] == "-1"
        # save the op converted rows
        results[ip_op_row_ix].groupby(["age", "sex", "tretspef"]).size().to_frame(
            "attendances"
        ).assign(tele_attendances=0).reset_index().to_parquet(
            f"{path_fn('op_conversion')}/0.parquet"
        )
        # remove the op converted rows
        results.loc[~ip_op_row_ix, ["speldur", "classpat"]].to_parquet(
            f"{path_fn('ip')}/0.parquet"
        )
