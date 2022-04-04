"""
Outpatients Module

Implements the Outpatients model.
"""

import numpy as np
import pandas as pd

from model.helpers import age_groups
from model.model import Model


class OutpatientsModel(Model):
    """
    Outpatients Model

    * results_path: where the data is stored

    Implements the model for outpatient data. See `Model()` for documentation on the generic class.
    """

    def __init__(self, results_path):
        # call the parent init function
        Model.__init__(self, "op", results_path)

    #
    def _followup_reduction(self, data, run_params):
        return self._factor_helper(
            data, run_params["followup_reduction"], {"has_procedures": 0, "is_first": 0}
        )

    #
    def _consultant_to_consultant_reduction(self, data, run_params):
        return self._factor_helper(
            data,
            run_params["consultant_to_consultant_reduction"],
            {"is_cons_cons_ref": 1},
        )

    #
    @staticmethod
    def _convert_to_tele(data, run_params):
        # temp disable chained assignment warnings
        options = pd.get_option("mode.chained_assignment")
        pd.set_option("mode.chained_assignment", None)
        # get the parameters
        params = run_params["convert_to_tele"]
        # find locations of rows that didn't have procedures
        npix = ~data["has_procedures"]
        # create a value for converting attendances into tele attendances for each row
        # the value will be a random binomial value, i.e. we will convert between 0 and attendances
        # into tele attendances
        tele_conversion = np.random.binomial(
            data.loc[npix, "attendances"], [params[t] for t in data.loc[npix, "type"]]
        )
        # update the columns, subtracting tc from one, adding tc to the other (we maintain the
        # number of overall attendances)
        data.loc[npix, "attendances"] -= tele_conversion
        data.loc[npix, "tele_attendances"] += tele_conversion
        # restore chained assignment warnings
        pd.set_option("mode.chained_assignment", options)

    #
    def _run(
        self, rng, data, run_params, aav_f, hsa_f
    ):  # pylint: disable=too-many-arguments
        """
        Run the model once

        returns: a tuple of the selected varient and the updated DataFrame
        """
        params = run_params["outpatient_factors"]
        #
        sc_a, sc_t = data[["attendances", "tele_attendances"]].sum()
        step_counts = {
            "baseline": pd.DataFrame(
                {"attendances": [sc_a], "tele_attendances": [sc_t]}, ["-"]
            )
        }

        def update_stepcounts(name):
            nonlocal data, step_counts, sc_a, sc_t
            # update the step count values
            sc_ap = sum(data["attendances"])
            sc_tp = sum(data["tele_attendances"])
            step_counts[name] = pd.DataFrame(
                {"attendances": [sc_ap - sc_a], "tele_attendances": [sc_tp - sc_t]},
                ["-"],
            )
            # replace the values
            sc_a, sc_t = sc_ap, sc_tp

        def run_step(factor, name):
            nonlocal data
            # perform the step
            data.loc[:, "attendances"] = rng.poisson(
                data["attendances"].to_numpy() * factor
            )
            data.loc[:, "tele_attendances"] = rng.poisson(
                data["tele_attendances"].to_numpy() * factor
            )
            # remove rows where the overall number of attendances was 0
            data = data[data["attendances"] + data["tele_attendances"] > 0]
            update_stepcounts(name)

        # captue current pandas options, and set chainged assignment off
        pd_options = pd.set_option("mode.chained_assignment", None)
        # before we do anything, reset the index to keep the row number
        data.reset_index(inplace=True)
        # first, run hsa as we have the factor already created
        run_step(hsa_f, "health_status_adjustment")
        # then, demographic modelling
        run_step(aav_f[data["rn"]], "population_factors")
        # now run strategies
        run_step(self._followup_reduction(data, params), "followup_reduction")
        run_step(
            self._consultant_to_consultant_reduction(data, params),
            "consultant_to_consultant_referrals",
        )
        # now run_step's are finished, restore pandas options
        pd.set_option("mode.chained_assignment", pd_options)
        # convert attendances to tele attendances
        self._convert_to_tele(data, params)
        update_stepcounts("tele_conversion")
        # return the data
        change_factors = pd.melt(
            pd.concat(step_counts)
            .rename_axis(["change_factor", "strategy"])
            .reset_index(),
            ["change_factor", "strategy"],
            ["attendances", "tele_attendances"],
            "measure",
        )
        change_factors["value"] = change_factors["value"].astype(int)
        return (change_factors, data.drop(["hsagrp"], axis="columns"))

    def aggregate(self, model_results):
        """
        Aggregate the model results
        """
        model_results["age_group"] = age_groups(model_results["age"])

        return self._aggregate_op_rows(model_results)

    @staticmethod
    def _aggregate_op_rows(op_rows):
        op_rows.loc[op_rows["is_first"], "pod"] = "op_first"
        op_rows.loc[~op_rows["is_first"], "pod"] = "op_follow-up"
        op_rows.loc[op_rows["has_procedures"], "pod"] = "op_procedure"

        op_agg = op_rows.groupby(
            ["age_group", "sex", "tretspef", "pod"],
            as_index=False,
        ).agg({"attendances": np.sum, "tele_attendances": np.sum})

        return pd.melt(
            op_agg,
            ["age_group", "sex", "tretspef", "pod"],
            ["attendances", "tele_attendances"],
            "measure",
        )
