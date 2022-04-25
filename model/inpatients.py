"""
Inpatients Module

Implements the inpatients model.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from model.helpers import age_groups, inrange
from model.model import Model


class InpatientsModel(Model):
    """
    Inpatients Model

    * results_path: where the data is stored

    Implements the model for inpatient data. See `Model()` for documentation on the generic class.
    """

    def __init__(self, results_path: str):
        # call the parent init function
        Model.__init__(
            self,
            "ip",
            results_path,
            [
                "rn",
                "speldur",
                "age",
                "sex",
                "admimeth",
                "classpat",
                "tretspef",
                "hsagrp",
            ],
        )
        # load the strategies, store each strategy file as a separate entry in a dictionary
        self._strategies = {
            x: self._load_parquet(f"ip_{x}_strategies").set_index(["rn"]).iloc[:, 0]
            for x in ["admission_avoidance", "los_reduction"]
        }

    #
    def _los_reduction(self, run_params):
        """
        Create a dictionary of the los reduction factors to use for a run

        * rng: an instance of np.random.default_rng, created for each model iteration
        """
        params = self._params["strategy_params"]["los_reduction"]
        # convert the parameters dictionary to a dataframe: each item becomes a row (with the item
        # being the name of the row in the index), and then each sub-item becoming a column
        losr = pd.DataFrame.from_dict(params, orient="index")
        losr["losr_f"] = [
            run_params["strategy_params"]["los_reduction"][i] for i in losr.index
        ]
        return losr

    #
    def _random_strategy(self, rng, strategy_type):
        """
        Select one strategy per record

        * rng: an instance of np.random.default_rng, created for each model iteration
        * strategy_type: a string of which type of strategy to update, e.g. "admission_avoidance",
          "los_reduction"

        returns: an updated DataFrame with a new column for the selected strategy
        """
        strategies = self._strategies[strategy_type]
        # first, filter the strategies to only include those listed in the params file
        valid_strategies = list(
            self._params["strategy_params"][strategy_type].keys()
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

    #
    def _waiting_list_adjustment(self, data):
        """
        Create a series of factors for waiting list adjustment.

        * data: the pandas DataFrame that we are updating

        returns: a series of floats indicating how often we want to sample that row

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline
        """
        # extract the waiting list adjustment parameters - we convert this to a default dictionary
        # that uses the "X01" specialty as the default value
        pwla = self._params["waiting_list_adjustment"]["inpatients"].copy()
        default_specialty = pwla.pop("X01")
        pwla = defaultdict(lambda: default_specialty, pwla)
        # waiting list adjustment values
        # create a series of 1's the length of our data: make sure these are floats, not ints
        wlav = np.ones_like(data.index).astype(float)
        # find the rows which are elective wait list admissions
        i = list(data.admimeth == "11")
        # update the series for these rows with the waiting list adjustments for that specialty
        wlav[i] = [pwla[t] for t in data[i].tretspef]
        # return the waiting list adjustment factor series
        return wlav

    @staticmethod
    def _admission_avoidance_step(rng, data, admission_avoidance, run_params):
        # choose admission avoidance factors
        ada = defaultdict(
            lambda: 1, run_params["strategy_params"]["admission_avoidance"]
        )
        # work out the number of rows/sum of los before admission avoidance
        data_aa = data.merge(
            admission_avoidance, left_on="rn", right_index=True
        ).groupby("admission_avoidance_strategy")
        sc_n_aa = data_aa["rn"].agg(len)
        sc_b_aa = (
            # add rows to convert los to beddays
            data_aa["speldur"].agg(sum)
            + sc_n_aa
        )
        # then, work out the admission avoidance factors for each row
        aaf = [ada[k] for k in admission_avoidance[data["rn"]]]
        # decide whether to select this row or not
        select_row = rng.binomial(n=1, p=aaf).astype(bool)
        # select each row as many times as it was sampeld
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
        return (
            data,
            pd.DataFrame(
                {
                    "admissions": {
                        k: sc_n_aa_post[k] - sc_n_aa[k] for k in sc_n_aa.index
                    },
                    "beddays": {k: sc_b_aa_post[k] - sc_b_aa[k] for k in sc_b_aa.index},
                }
            ),
        )

    #
    @staticmethod
    def _losr_all(data, losr, rng, step_counts):
        """
        Length of Stay Reduction: All

        * data: the pandas DataFrame that we are updating
        * losr: the Length of Stay rates table created from self._los_reduction()
        * rng: an instance of np.random.default_rng, created for each model iteration

        returns: a dataframe with an updated length of stay column

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
        )
        step_counts["los_reduction"] = {
            "admissions": (change_los * 0).to_dict(),
            "beddays": change_los.to_dict(),
        }

    @staticmethod
    def _losr_bads(data, losr, rng, step_counts):
        """
        Length of Stay Reduction: British Association of Day Surgery

        * data: the pandas DataFrame that we are updating
        * losr: the Length of Stay rates table created from self._los_reduction()
        * rng: an instance of np.random.default_rng, created for each model iteration

        returns: a dataframe with an updated patient classification and length of stay column's

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
        data.loc[i, "speldur"] *= data.loc[i, "classpat"] == 1
        #
        step_counts["los_reduction"]["admissions"] = {
            **step_counts["los_reduction"]["admissions"],
            **((data.loc[i, "classpat"] == "-1").groupby(level=0).sum() * -1)
            .astype(int)
            .to_dict(),
        }
        step_counts["los_reduction"]["beddays"] = {
            **step_counts["los_reduction"]["beddays"],
            **(data.loc[i, "speldur"] - bads_df["speldur"])
            .groupby(level=0)
            .sum()
            .astype(int)
            .to_dict(),
        }

    @staticmethod
    def _losr_to_zero(data, losr, rng, losr_type, step_counts):
        """
        Length of Stay Reduction: To Zero Day LoS

        * data: the pandas DataFrame that we are updating
        * losr: the Length of Stay rates table created from self._los_reduction()
        * rng: an instance of np.random.default_rng, created for each model iteration
        * type: the type of row we are updating

        returns: a dataframe with an updated length of stay column

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
        )
        step_counts["los_reduction"]["admissions"] = {
            **step_counts["los_reduction"]["admissions"],
            **(change_los * 0).to_dict(),
        }
        step_counts["los_reduction"]["beddays"] = {
            **step_counts["los_reduction"]["beddays"],
            **change_los.to_dict(),
        }

    #
    def _run(
        self, rng, data, run_params, aav_f, hsa_f
    ):  # pylint: disable=too-many-arguments
        # select strategies
        admission_avoidance = self._random_strategy(rng, "admission_avoidance")
        los_reduction = self._random_strategy(rng, "los_reduction")
        # choose length of stay reduction factors
        losr = self._los_reduction(run_params)
        #
        sc_n, sc_b = len(data.index), sum(data["speldur"] + 1)
        step_counts = {
            "baseline": pd.DataFrame({"admissions": [sc_n], "beddays": [sc_b]}, ["-"])
        }
        #
        def run_step(thing, name):
            nonlocal data, step_counts, sc_n, sc_b
            select_row_n_times = rng.poisson(thing)
            # perform the step
            data = data.loc[data.index.repeat(select_row_n_times)].reset_index(
                drop=True
            )
            # update the step count values
            sc_np = int(sum(select_row_n_times))
            sc_bp = int(sum(data["speldur"] + 1))
            step_counts[name] = pd.DataFrame(
                {"admissions": [sc_np - sc_n], "beddays": [sc_bp - sc_b]}, ["-"]
            )
            # replace the values
            sc_n, sc_b = sc_np, sc_bp

        # before we do anything, reset the index to keep the row number
        data.reset_index(inplace=True)
        # first, run hsa as we have the factor already created
        run_step(hsa_f, "health_status_adjustment")
        # then, demographic modelling
        run_step(aav_f[data["rn"]], "population_factors")
        # waiting list adjustments
        run_step(self._waiting_list_adjustment(data), "waiting_list_adjustment")
        # Admission Avoidance ----------------------------------------------------------------------
        data, step_counts["admission_avoidance"] = self._admission_avoidance_step(
            rng, data, admission_avoidance, run_params
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
        step_counts["los_reduction"] = pd.DataFrame(step_counts["los_reduction"])
        # return the data (select just the columns we have updated in modelling)
        change_factors = pd.melt(
            pd.concat(step_counts)
            .rename_axis(["change_factor", "strategy"])
            .reset_index(),
            ["change_factor", "strategy"],
            ["admissions", "beddays"],
            "measure",
        )
        change_factors["value"] = change_factors["value"].astype(int)
        return (change_factors, data.drop(["hsagrp"], axis="columns").set_index(["rn"]))

    def aggregate(self, model_results):
        """
        Aggregate the model results
        """
        model_results["age_group"] = age_groups(model_results["age"])
        # find the rows we need to convert to outpatients
        ip_op_row_ix = model_results["classpat"] == "-1"
        return pd.concat(
            [
                self._aggregate_ip_rows(model_results[~ip_op_row_ix]),
                self._aggregate_op_rows(model_results[ip_op_row_ix]),
            ]
        )

    @staticmethod
    def _aggregate_op_rows(op_rows):
        op_rows = op_rows.copy()
        return (
            op_rows.value_counts(["age_group", "sex", "tretspef"])
            .to_frame("value")
            .reset_index()
            .assign(pod="op_procedure", measure="attendances")
        )

    @staticmethod
    def _aggregate_ip_rows(ip_rows):
        ip_rows = ip_rows.copy()
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

        ip_agg = (
            ip_rows.groupby(
                ["age_group", "sex", "tretspef", "pod"],
                as_index=False,
            )
            .agg({"speldur": len, "beddays": np.sum})
            .rename({"speldur": "admissions"}, axis="columns")
        )

        return pd.melt(
            ip_agg,
            ["age_group", "sex", "tretspef", "pod"],
            ["admissions", "beddays"],
            "measure",
        )
