import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from model.helpers import inrange, rnorm
from model.Model import Model


class OutpatientsModel(Model):
    """
    Outpatients Model

    Implements the model for outpatient data. See `Model()` for documentation on the generic class.
    """

    def __init__(self, results_path):
        self._MODEL_TYPE = "op"
        # call the parent init function
        Model.__init__(self, results_path)

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
    def _convert_to_tele(self, data, run_params):
        # temp disable chained assignment warnings
        o = pd.get_option("mode.chained_assignment")
        pd.set_option("mode.chained_assignment", None)
        # create a value for converting attendances into tele attendances for each row
        # the value will be a random binomial value, i.e. we will convert between 0 and attendances into tele attendances
        # find locations of rows that didn't have procedures
        npix = ~data["has_procedures"]
        p = run_params["convert_to_tele"]
        tc = np.random.binomial(
            data.loc[npix, "attendances"], [p[t] for t in data.loc[npix, "type"]]
        )
        # update the columns, subtracting tc from one, adding tc to the other (we maintain the number of overall attendances)
        data.loc[npix, "attendances"] -= tc
        data.loc[npix, "tele_attendances"] += tc
        # restore chained assignment warnings
        pd.set_option("mode.chained_assignment", o)

    #
    def _run(self, rng, data, run_params, hsa_f):
        """
        Run the model once

        returns: a tuple of the selected varient and the updated DataFrame
        """
        p = run_params["outpatient_factors"]
        # create a single factor for how many times to select that row
        factor = (
            data["factor"].to_numpy()
            * hsa_f
            * self._followup_reduction(data, p)
            * self._consultant_to_consultant_reduction(data, p)
        )
        # update the number of attendances / tele_attendances
        data["attendances"] = rng.poisson(data["attendances"] * factor)
        data["tele_attendances"] = rng.poisson(data["tele_attendances"] * factor)
        # remove rows where the overall number of attendances was 0
        data = data[data["attendances"] + data["tele_attendances"] > 0]
        # convert attendances to tele attendances
        self._convert_to_tele(data, p)
        # return the data
        return data[["attendances", "tele_attendances"]].reset_index()
