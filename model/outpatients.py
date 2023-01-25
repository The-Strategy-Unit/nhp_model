"""
Outpatients Module

Implements the Outpatients model.
"""

import os
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from model.model import Model


class OutpatientsModel(Model):
    """
    Outpatients Model

    Implementation of the Model for Outpatient attendances.

    :param params: the parameters to run the model with
    :param data_path: the path to where the data files live
    """

    def __init__(self, params: list, data_path: str) -> None:
        # call the parent init function
        super().__init__("op", params, data_path)

    def _waiting_list_adjustment(self, data: pd.DataFrame) -> pd.Series:
        """Create a series of factors for waiting list adjustment.

        A value of 1 will indicate that we want to sample this row at the baseline rate. A value
        less that 1 will indicate we want to sample that row less often that in the baseline, and
        a value greater than 1 will indicate that we want to sample that row more often than in the
        baseline

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame

        :returns: an array of factors, the length of data
        :rtype: numpy.ndarray
        """
        tretspef_n = data.groupby("tretspef").size()
        wla_param = pd.Series(self.params["waiting_list_adjustment"]["op"])
        wla = (wla_param / tretspef_n).fillna(0) + 1
        return wla[data.tretspef].to_numpy()

    def _followup_reduction(self, data: pd.DataFrame, run_params: dict) -> np.ndarray:
        """Followup Appointment Reduction

        Returns the factor values for the Followup Appointment Reduction strategy

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: an array of factors, the length of data
        :rtype: numpy.ndarray
        """
        return self._factor_helper(
            data, run_params["followup_reduction"], {"has_procedures": 0, "is_first": 0}
        )

    def _consultant_to_consultant_reduction(
        self, data: pd.DataFrame, run_params: dict
    ) -> np.ndarray:
        """Consultant to Consultant Referral Reduction


        Returns the factor values for the Consultant to Consultant Reduction strategy

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: an array of factors, the length of data
        :rtype: numpy.ndarray
        """
        return self._factor_helper(
            data,
            run_params["consultant_to_consultant_reduction"],
            {"is_cons_cons_ref": 1},
        )

    @staticmethod
    def _convert_to_tele(
        rng: np.random.Generator,
        data: pd.DataFrame,
        run_params: dict,
        step_counts: dict,
    ) -> None:
        """Convert attendances to tele-attendances

        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict
        """
        # get the parameters
        params = run_params["convert_to_tele"]
        # find locations of rows that didn't have procedures
        npix = ~data["has_procedures"]
        # create a value for converting attendances into tele attendances for each row
        # the value will be a random binomial value, i.e. we will convert between 0 and attendances
        # into tele attendances
        tele_conversion = rng.binomial(
            data.loc[npix, "attendances"], [params[t] for t in data.loc[npix, "type"]]
        )
        # update the columns, subtracting tc from one, adding tc to the other (we maintain the
        # number of overall attendances)
        data.loc[npix, "attendances"] -= tele_conversion
        data.loc[npix, "tele_attendances"] += tele_conversion
        tele_conversion = tele_conversion.sum()
        step_counts["convert_to_tele"] = {
            "attendances": -tele_conversion,
            "tele_attendances": tele_conversion,
        }

    @staticmethod
    def _expat_adjustment(data: pd.DataFrame, run_params: dict) -> pd.Series:
        expat_params = run_params["expat"]["op"]
        join_cols = ["tretspef"]
        return data.merge(
            pd.DataFrame(
                [{"tretspef": k, "value": v} for k, v in expat_params.items()]
            ).set_index(join_cols),
            how="left",
            left_on=join_cols,
            right_index=True,
        )["value"].fillna(1)

    @staticmethod
    def _repat_adjustment(data: pd.DataFrame, run_params: dict) -> pd.Series:
        repat_local_params = run_params["repat_local"]["op"]
        repat_nonlocal_params = run_params["repat_nonlocal"]["op"]
        join_cols = ["tretspef", "is_main_icb"]
        return data.merge(
            pd.DataFrame(
                [
                    {"tretspef": k1, "is_main_icb": icb, "value": v1}
                    for (k0, icb) in [
                        (repat_local_params, True),
                        (repat_nonlocal_params, False),
                    ]
                    for k1, v1 in k0.items()
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

        # create an attendance type series: procedure should overwrite first/followup
        attend_type = pd.Series("first", index=data.index)
        attend_type.loc[~data["is_first"]] = "followup"
        attend_type.loc[data["has_procedures"]] = "procedure"

        # extract the parameters
        params = run_params["baseline_adjustment"]["op"]

        # get the parameter value for each row
        return np.array([params[k][t] for (k, t) in zip(attend_type, data["tretspef"])])

    @staticmethod
    def _step_counts(
        attendances: pd.Series,
        tele_attendances: pd.Series,
        attendances_after: int,
        tele_attendances_after: int,
        factors: dict,
    ) -> dict:
        cf = {}
        # convert the attendances/tele_attendances before modelling to a complex number:
        #   * the "real" part is the attendances count
        #   * the "imaginary" part is the tele attendances count
        # this makes it easy to keep track of both values
        n = (attendances + tele_attendances * 1j).to_numpy()
        before = ns = n.sum()
        for k, v in factors.items():
            n2 = n * v
            n2s = n2.sum()
            cf[k] = n2s - ns
            n, ns = n2, n2s

        attendances_slack = 1 + (attendances_after - ns.real) / (ns.real - before.real)
        tele_attendances_slack = 1 + (tele_attendances_after - ns.imag) / (
            ns.imag - before.imag
        )
        return {
            "baseline": {"attendances": before.real, "tele_attendances": before.imag},
            **{
                k: {
                    "attendances": v.real * attendances_slack,
                    "tele_attendances": v.imag * tele_attendances_slack,
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
        params = run_params["outpatient_factors"]

        factors = {
            "health_status_adjustment": hsa_f,
            "population_factors": demo_f[data["rn"]].to_numpy(),
            "expatriation": self._expat_adjustment(data, run_params).to_numpy(),
            "repatriation": self._repat_adjustment(data, run_params).to_numpy(),
            "baseline_adjustment": self._baseline_adjustment(data, run_params),
            "waiting_list_adjustment": self._waiting_list_adjustment(data),
            "followup_reduction": self._followup_reduction(data, params),
            "consultant_to_consultant_referrals": self._consultant_to_consultant_reduction(
                data, params
            ),
        }
        reduced_factor = np.prod(list(factors.values()), axis=0)

        new_attendances = rng.poisson(data["attendances"] * reduced_factor)
        new_tele_attendances = rng.poisson(data["tele_attendances"] * reduced_factor)

        step_counts = self._step_counts(
            data["attendances"],
            data["tele_attendances"],
            new_attendances.sum(),
            new_tele_attendances.sum(),
            factors,
        )

        data["attendances"] = new_attendances
        data["tele_attendances"] = new_tele_attendances

        # convert attendances to tele attendances
        self._convert_to_tele(rng, data, params, step_counts)

        # return the data
        change_factors = (
            pd.DataFrame.from_dict(step_counts, orient="index")
            .rename_axis("change_factor")
            .reset_index()
            .assign(strategy="-")
            .melt(
                ["change_factor", "strategy"],
                ["attendances", "tele_attendances"],
                "measure",
            )
        )
        return (change_factors, data.drop(["hsagrp"], axis="columns"))

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
        model_results.loc[model_results["is_first"], "pod"] = "op_first"
        model_results.loc[~model_results["is_first"], "pod"] = "op_follow-up"
        model_results.loc[model_results["has_procedures"], "pod"] = "op_procedure"

        measures = model_results.melt(
            ["rn"], ["attendances", "tele_attendances"], "measure"
        )
        model_results = (
            model_results.drop(["attendances", "tele_attendances"], axis="columns")
            .merge(measures, on="rn")
            # summarise the results to make the create_agg steps quicker
            .groupby(
                # note: any columns used in the calls to _create_agg, including pod and measure
                # must be included below
                ["pod", "measure", "sex", "age_group", "tretspef"],
                as_index=False,
            )
            .agg({"value": "sum"})
        )

        agg = partial(self._create_agg, model_results)
        return {
            **agg(),
            **agg(["sex", "age_group"]),
            **agg(["sex", "tretspef"]),
        }

    def save_results(
        self, model_results: pd.DataFrame, path_fn: Callable[[str], str]
    ) -> None:
        """Save the results of running the model

        This method is used for saving the results of the model run to disk as a parquet file.
        It saves just the `rn` (row number) column and the `arrivals`, with the intention that
        you rejoin to the original data.

        :param model_results: a DataFrame containing the results of a model iteration
        :param path_fn: a function which takes the activity type and returns a path
        """
        model_results.set_index(["rn"])[["attendances", "tele_attendances"]].to_parquet(
            f"{path_fn('op')}/0.parquet"
        )
