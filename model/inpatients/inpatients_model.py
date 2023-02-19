"""
Inpatients Module

Implements the inpatients model.
"""
import json
from collections import namedtuple
from datetime import timedelta
from functools import partial
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.activity_avoidance import ActivityAvoidance
from model.inpatients.los_reduction import los_reduction
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

    def __init__(self, params: list, data_path: str) -> None:
        # call the parent init function
        super().__init__(
            "ip",
            params,
            data_path,
            columns_to_load=[
                "rn",
                "sitetret",
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
        self.data = self.data.assign(is_wla=lambda x: x.admimeth == "11").rename(
            columns={"admigroup": "group"}
        )
        # load the strategies, store each strategy file as a separate entry in a dictionary
        self._strategies = {
            x: self._load_strategies(x)
            for x in ["admission_avoidance", "los_reduction"]
        }
        # load the theatres data
        self._load_theatres_data()
        # load the kh03 data
        self._load_kh03_data()
        # oversample
        self._union_bedday_rows()
        # make sure to create the counts after oversampling
        self._baseline_counts = self._get_data_counts(self.data)
        # TODO: this should be fixed by renaming files and altering the columns
        self._strategies["activity_avoidance"] = self._strategies.pop(
            "admission_avoidance"
        ).rename(columns={"admission_avoidance_strategy": "strategy"})

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
        # create a row mask for use when counting rows
        self._data_mask = ~self.data["bedday_rows"].to_numpy()

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

    def _load_strategies(self, strategy_type: str) -> pd.DataFrame:
        """Load a set of strategies

        :param strategy_type: either admission_avoidance or los_reduction
        :type strategy_type: str
        :return: a dataframe containing the strategies
        :rtype: _type_
        """
        # load the file
        s = self._load_parquet(f"ip_{strategy_type}_strategies").set_index(["rn"])
        # get the valid set of valid strategies from the params
        p = self.params["inpatient_factors"][strategy_type].keys()
        # subset the strategies
        return s[s.iloc[:, 0].isin(p)]

    def _get_data_counts(self, data: pd.DataFrame) -> npt.ArrayLike:
        return np.array(
            [np.ones_like(data["rn"]), (1 + data["speldur"]).to_numpy()]
        ).astype(float)

    def _apply_resampling(self, row_samples, data):
        data = data.loc[data.index.repeat(row_samples[0])].reset_index(drop=True)
        # filter out the "oversampled" rows from the counts
        mask = (~data["bedday_rows"]).to_numpy().astype(float)
        # return the altered data and the amount of admissions/beddays after resampling
        return (data, self._get_data_counts(data) * mask)

    def _run(self, model_run: ModelRun) -> tuple[dict, pd.DataFrame]:
        """Run the model once

        Performs a single iteration of the model. The `model_run` parameter controls what parameters
        to use for this run of the model, with the principal model run having 0, and all other model
        runs having a value greater than 0.

        Running the model multiple times with the same `model_run` will give the same results.

        The return value is a tuple that contains the change factors as a dictionary, and the model
        results as a :class:`pandas.DataFrame`.

        :param model_run: the current model run
        :type model_run: int
        :returns: a tuple of the change factors and the model results
        :rtype: (dict, pandas.DataFrame)
        """

        data, step_counts = (
            ActivityAvoidance(
                model_run, self._baseline_counts, row_mask=self._data_mask
            )
            .demographic_adjustment()
            .health_status_adjustment()
            .expat_adjustment()
            .repat_adjustment()
            .baseline_adjustment()
            .waiting_list_adjustment()
            .non_demographic_adjustment()
            .activity_avoidance()
            .apply_resampling()
        )

        step_counts = {
            k: {"admissions": v[0], "beddays": v[1]} for k, v in step_counts.items()
        }

        # los reduction
        los_reduction(model_run, data, self._strategies["los_reduction"], step_counts)
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
            .agg({"size": np.mean})
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

        # sum the amount of procedures, by specialty,
        # then keep only the ones which appear in fhs_baseline
        future = (
            model_results[model_results["measure"] == "procedures"]
            .groupby("tretspef")["value"]
            .sum()
        )
        future = future[future.index.isin(fhs_baseline.index)]

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
                    "sitetret",
                    "age_group",
                    "sex",
                    "admimeth",
                    "group",
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
