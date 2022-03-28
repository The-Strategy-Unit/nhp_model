from collections import defaultdict

import numpy as np
import pandas as pd

from model.helpers import inrange, rnorm
from model.Model import Model


class InpatientsModel(Model):
    """
    Inpatients Model

    Implements the model for inpatient data. See `Model()` for documentation on the generic class.
    """

    def __init__(self, results_path):
        self._MODEL_TYPE = "ip"
        # call the parent init function
        Model.__init__(
            self,
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
            x: self._load_parquet(f"ip_{x}_strategies")
            for x in ["admission_avoidance", "los_reduction"]
        }

    #
    def _los_reduction(self, run_params):
        """
        Create a dictionary of the los reduction factors to use for a run

        * rng: an instance of np.random.default_rng, created for each model iteration
        """
        p = self._params["strategy_params"]["los_reduction"]
        # convert the parameters dictionary to a dataframe: each item becomes a row (with the item being the name of the
        # row in the index), and then each sub-item becoming a column
        losr = pd.DataFrame.from_dict(p, orient="index")
        losr["losr_f"] = [
            run_params["strategy_params"]["los_reduction"][i] for i in losr.index
        ]
        return losr

    #
    def _random_strategy(self, rng, data, strategy_type):
        """
        Select one strategy per record

        * rng: an instance of np.random.default_rng, created for each model iteration
        * data: the pandas DataFrame that we are updating
        * strategy_type: a string of which type of strategy to update, e.g. "admission_avoidance", "los_reduction"

        returns: an updated DataFrame with a new column for the selected strategy
        """
        df = (
            self._strategies[strategy_type]
            # take all of the rows and randomly reshuffle them into a new order. We *do not* want to use resampling here.
            # make sure to use the same random state using rng.bit_generator
            .sample(frac=1, random_state=rng.bit_generator)
            # for each rn, select a single row, i.e. select 1 strategy per rn
            .groupby(["rn"]).head(1)
            # set the index to match the data objects index
            .set_index(["rn"])
        )
        # return the data joined to the selected strategies
        return data.merge(df, left_index=True, right_index=True)

    #
    def _waiting_list_adjustment(self, data):
        """
        Create a series of factors for waiting list adjustment.

        * data: the pandas DataFrame that we are updating

        returns: a series of floats indicating how often we want to sample that row

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value less that 1 will indicate
        we want to sample that row less often that in the baseline, and a value greater than 1 will indicate that we want
        to sample that row more often than in the baseline
        """
        # extract the waiting list adjustment parameters - we convert this to a default dictionary that uses the "X01"
        # specialty as the default value
        pwla = self._params["waiting_list_adjustment"]["inpatients"].copy()
        dv = pwla.pop("X01")
        pwla = defaultdict(lambda: dv, pwla)
        # create a series of 1's the length of our data: make sure these are floats, not ints
        w = np.ones_like(data.index).astype(float)
        # find the rows which are elective wait list admissions
        i = list(data.admimeth == "11")
        # update the series for these rows with the waiting list adjustments for that specialty
        w[i] = [pwla[t] for t in data[i].tretspef]
        # return the waiting list adjustment factor series
        return w

    #
    def _losr_all(self, data, losr, rng):
        """
        Length of Stay Reduction: All

        * data: the pandas DataFrame that we are updating
        * losr: the Length of Stay rates table created from self._los_reduction()
        * rng: an instance of np.random.default_rng, created for each model iteration

        returns: a dataframe with an updated length of stay column

        Reduces all rows length of stay by sampling from a binomial distribution, using the current length of stay as the
        value for n, and the length of stay reduction factor for that strategy as the value for p. This will update the
        los to be a value between 0 and the original los.
        """
        s = losr.index[losr.type == "all"]
        data.loc[s, "speldur"] = rng.binomial(
            data.loc[s, "speldur"], losr.loc[data.loc[s].index, "losr_f"]
        )
        return data

    def _losr_bads(self, data, losr, rng):
        """
        Length of Stay Reduction: British Association of Day Surgery

        * data: the pandas DataFrame that we are updating
        * losr: the Length of Stay rates table created from self._los_reduction()
        * rng: an instance of np.random.default_rng, created for each model iteration

        returns: a dataframe with an updated patient classification and length of stay column's

        This will swap rows between elective admissions and daycases into either daycases or outpatients, based on the given
        parameter values. We have a baseline target rate, this is the rate in the baseline that rows are of the given target
        type (i.e. either daycase, outpatients, daycase or outpatients). Our interval will alter the target_rate by setting
        some rows which are not the target type to be the target type.

        Rows that are converted to daycase have the patient classification set to 2. Rows that are converted to outpatients
        have a patient classification of -1 (these rows need to be filtered out of the inpatients results and added to the
        outpatients results).

        Rows that are modelled away from elective care have the length of stay fixed to 0 days.
        """
        i = losr.type == "bads"
        # create a copy of our data, and join to the losr data
        bads_df = data.merge(
            losr[i][["baseline_target_rate", "op_dc_split", "losr_f"]],
            left_index=True,
            right_index=True,
        )
        # convert the factor value to be the amount we need to adjust the non-target type to to make the target rate equal
        # too the factor value
        r = (
            (bads_df["losr_f"] - bads_df["baseline_target_rate"])
            / (1 - bads_df["baseline_target_rate"])
        ).apply(inrange)
        # create three values that will sum to 1 - these will be the probabilties of:
        #   - staying where we are  [0.0, u0)
        #   - moving to daycase     [u0,  u1)
        #   - moving to outpatients (u1,  1.0]
        u1 = bads_df["op_dc_split"] * r
        u2 = (1 - bads_df["op_dc_split"]) * r
        u0 = 1 - u1 - u2
        # we now create a random value for each row in [0, 1]
        x = rng.uniform(size=len(r))
        # we can use this random value to update the patient class column appropriately
        bads_df.loc[(x >= u0) & (x < u0 + u1), "classpat"] = "2"  # row becomes daycase
        bads_df.loc[(x >= u0 + u1), "classpat"] = "-1"  # row becomes outpatients
        # we now need to apply these changes to the actual data
        s = losr.index[i]
        data.loc[s, "classpat"] = bads_df["classpat"]
        data.loc[s, "speldur"] *= (
            data.loc[s, "classpat"] == 1
        )  # set the speldur to 0 if we aren't inpatients
        return data

    def _losr_to_zero(self, data, losr, rng, type):
        """
        Length of Stay Reduction: To Zero Day LoS

        * data: the pandas DataFrame that we are updating
        * losr: the Length of Stay rates table created from self._los_reduction()
        * rng: an instance of np.random.default_rng, created for each model iteration
        * type: the type of row we are updating

        returns: a dataframe with an updated length of stay column

        Updates the length of stay to 0 for a given percentage of rows.
        """
        s = losr.index[losr.type == type]
        n = len(data.loc[s, "speldur"])
        data.loc[s, "speldur"] *= (
            rng.uniform(size=n) >= losr.loc[data.loc[s].index, "losr_f"]
        )
        return data

    #
    def _run(self, rng, data, run_params, hsa_f):
        """
        Run the model once

        * model_run: the number of the model to run. This set's the random seed so the results are reproducible

        returns: a tuple of the selected varient and the updated DataFrame
        """
        # choose admission avoidance factors
        ada = defaultdict(
            lambda: 1, run_params["strategy_params"]["admission_avoidance"]
        )
        # choose length of stay reduction factors
        losr = self._los_reduction(run_params)
        # select a strategy
        row_count = len(data.index)  # for assert below
        data = self._random_strategy(rng, data, "admission_avoidance")
        data = self._random_strategy(rng, data, "los_reduction")
        # double check that joining in the strategies didn't drop any rows
        assert (
            len(data.index) == row_count
        ), "Row's lost when selecting strategies: has the NULL strategy not been included?"
        # Admission Avoidance ----------------------------------------------------------------------------------------------
        factor_a = np.array([ada[k] for k in data["admission_avoidance_strategy"]])
        # waiting list adjustments
        factor_w = self._waiting_list_adjustment(data)
        # create a single factor for how many times to select that row
        n = rng.poisson(data["factor"].to_numpy() * hsa_f * factor_a * factor_w)
        # drop columns we don't need and repeat rows n times
        data = data.loc[data.index.repeat(n)].drop(["factor"], axis="columns")
        data.reset_index(inplace=True)
        # LoS Reduction ----------------------------------------------------------------------------------------------------
        # set the index for easier querying
        data.set_index(["los_reduction_strategy"], inplace=True)
        # run each of the length of stay reduction strategies
        data = self._losr_all(data, losr, rng)
        data = self._losr_to_zero(data, losr, rng, "aec")
        data = self._losr_to_zero(data, losr, rng, "preop")
        data = self._losr_bads(data, losr, rng)
        # return the data (select just the columns we have updated in modelling)
        return data.reset_index()[
            [
                "rn",
                "speldur",
                "classpat",
                "admission_avoidance_strategy",
                "los_reduction_strategy",
            ]
        ]

    #
    def _principal_projection(self, rng, data, run_params, hsa_f):
        # choose admission avoidance factors
        ada = defaultdict(
            lambda: 1, run_params["strategy_params"]["admission_avoidance"]
        )
        # choose length of stay reduction factors
        losr = self._los_reduction(run_params)
        # select a strategy
        data = self._random_strategy(rng, data, "admission_avoidance")
        data = self._random_strategy(rng, data, "los_reduction")
        #
        step_counts = {"baseline": len(data.index)}
        #
        def f(thing, name):
            nonlocal data, step_counts
            n = rng.poisson(thing)
            step_counts[name] = sum(n) - len(data.index)
            data = data.loc[data.index.repeat(n)].reset_index(drop=True)

        # before we do anything, reset the index to keep the row number
        data.reset_index(inplace=True)
        # first, run hsa as we have the factor already created
        f(hsa_f, "health_status_adjustment")
        # then, demographic modelling
        f(data["factor"].to_numpy(), "population_factors")
        data.drop(["factor"], axis="columns", inplace=True)
        # waiting list adjustments
        f(self._waiting_list_adjustment(data), "waiting_list_adjustment")
        # Admission Avoidance ----------------------------------------------------------------------------------------------
        aac = -data.value_counts("admission_avoidance_strategy")
        a = [ada[k] for k in data["admission_avoidance_strategy"]]
        n = rng.poisson(a)
        step_counts["admission_avoidance"] = sum(n)
        data = data.loc[data.index.repeat(n)].reset_index(drop=True)
        aac += data.value_counts("admission_avoidance_strategy")
        step_counts["admission_avoidance"] = aac.to_dict()
        # done with row count adjustment
        # LoS Reduction ----------------------------------------------------------------------------------------------------
        # set the index for easier querying
        data.set_index(["los_reduction_strategy"], inplace=True)
        # run each of the length of stay reduction strategies
        data = self._losr_all(data, losr, rng)
        data = self._losr_to_zero(data, losr, rng, "aec")
        data = self._losr_to_zero(data, losr, rng, "preop")
        data = self._losr_bads(data, losr, rng)
        # return the data (select just the columns we have updated in modelling)
        return (
            step_counts,
            data.reset_index()[
                [
                    "rn",
                    "speldur",
                    "classpat",
                    "admission_avoidance_strategy",
                    "los_reduction_strategy",
                ]
            ],
        )
