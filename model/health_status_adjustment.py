"""_summary_"""
# pylint: disable=too-few-public-methods

import json
from math import pi, sqrt

import numpy as np
import pandas as pd
import scipy.stats as spt


class HealthStatusAdjustment:
    """Health Status Adjustment

    handles the logic for the health status adjustment in the model"""

    # static variable for where data is stored
    data_path = "data"

    def __init__(self, data_path: str, base_year: str, model_year: str):
        self._all_ages = np.arange(0, 101)
        self._activity_table_path = f"{data_path}/hsa_activity_table.csv"

        self._load_life_expectancy_series(base_year, model_year)
        self._load_variant_lookup()
        self._load_activity_ages()
        self._cache = {}

    @staticmethod
    def _reference_path():
        return f"{HealthStatusAdjustment.data_path}/reference"

    def _load_life_expectancy_series(self, base_year: str, model_year: str):
        # the age range that health status adjustment runs for
        # hardcoded to max out at 90 as ages >90 are mapped to 90
        self._ages = np.arange(55, 91)
        # load the life expectancy file, only select the rows for the ages we are interested in
        lexc = (
            pd.read_csv(
                f"{HealthStatusAdjustment._reference_path()}/life_expectancy.csv"
            )
            .set_index(["var", "sex", "age"])
            .loc[slice(None), slice(None), self._ages]
        )
        # calculate the life expectancy (change) between the model year and base year
        self._life_expectancy = lexc[str(model_year)] - lexc[str(base_year)]

    def _load_variant_lookup(self):
        with open(
            f"{HealthStatusAdjustment._reference_path()}/variant_lookup.json",
            "r",
            encoding="UTF-8",
        ) as vlup_file:
            self._variant_lookup = json.load(vlup_file)

    def _load_activity_ages(self):
        self._activity_ages = (
            pd.read_csv(self._activity_table_path).set_index(["hsagrp", "sex", "age"])
        )["activity"]

    @staticmethod
    def generate_params(
        year: int, rng: np.random.BitGenerator, model_runs: int
    ) -> np.array:
        """Generate Health Status Adjustment Parameters

        :param year: The year the model is running for
        :type year: int
        :param rng: Random Number Generator
        :type rng: np.random.BitGenerator
        :param model_runs: Number of Model Runs
        :type model_runs: int
        :return: parameters for the health status adjustment
        :rtype: np.array
        """

        mode, sd1, sd2 = (
            pd.read_csv(
                f"{HealthStatusAdjustment._reference_path()}/hsa_split_normal_params.csv"
            )
            .set_index("year")
            .loc[year]
        )
        return [mode] + HealthStatusAdjustment.random_splitnorm(
            rng, model_runs, mode, sd1, sd2
        )

    @staticmethod
    def random_splitnorm(
        rng: np.random.BitGenerator,
        n: int,  # pylint: disable=invalid-name
        mode: float,
        sd1: float,
        sd2: float,
    ) -> np.array:
        # pylint: disable=invalid-name
        """Generate random splitnormal values

        :param rng: Random Number Generator
        :type rng: np.random.BitGenerator
        :param n: Number of random values to generate
        :type n: int
        :param mode: the mode of the distribution
        :type mode: float
        :param sd1: the standard deviation of the left side of the distribution
        :type sd1: float
        :param sd2: the standard deviation of the right side of the distribution
        :type sd2: float
        :return: n random number values sampled from the split normal distribution
        :rtype: np.array
        """
        # get the probability of the mode
        A = sqrt(2 / pi) / (sd1 + sd2)
        a_sqrt_tau = A * sqrt(2 * pi)
        p = (a_sqrt_tau * sd1) / 2

        # generate n random uniform values
        u = rng.uniform(size=n)

        # whether u is less than the mode or not
        a1 = u <= p

        # make a single sd vector
        sd = np.array([sd1 if i else sd2 for i in a1])
        x = np.array([0 if i else a_sqrt_tau * sd2 - 1 for i in a1])

        return mode + sd * spt.norm.ppf((u + x) / (a_sqrt_tau * sd))

    def run(self, run_params: dict):
        """_summary_

        :param hsa_param: _description_
        :type hsa_param: int
        :return: _description_
        :rtype: _type_
        """
        hsa_param = run_params["health_status_adjustment"]
        selected_variant = self._variant_lookup[run_params["variant"]]
        cache_key = (hsa_param, selected_variant)
        if cache_key in self._cache:
            return self._cache[cache_key]

        lexc = self._life_expectancy.loc[(selected_variant, slice(None), slice(None))]
        adjusted_ages = np.tile(self._ages, 2) - lexc * hsa_param

        self._cache[cache_key] = factor = (
            self._predict_activity(adjusted_ages).rename_axis(["hsagrp", "sex", "age"])
            / self._activity_ages.loc[slice(None), slice(None), self._ages]
        ).rename("health_status_adjustment")

        return factor

    def _predict_activity(self, adjusted_ages):
        """"""


class HealthStatusAdjustmentGAM(HealthStatusAdjustment):
    """_summary_"""

    def __init__(self, data_path: str, base_year: str, model_year: str):
        import pickle  # pylint: disable=import-outside-toplevel

        with open(f"{data_path}/hsa_gams.pkl", "rb") as hsa_pkl:
            self._gams = pickle.load(hsa_pkl)

        super().__init__(data_path, base_year, model_year)

    def _predict_activity(self, adjusted_ages):
        return pd.concat(
            {
                (h, s): pd.Series(
                    g.predict(adjusted_ages.loc[s]),
                    index=self._ages,
                )
                for (h, s), g in self._gams.items()
            }
        )


class HealthStatusAdjustmentInterpolated(HealthStatusAdjustment):
    """_summary_"""

    def __init__(self, data_path: str, base_year: str, model_year: str):
        super().__init__(data_path, base_year, model_year)
        self._load_activity_ages_lists()

    def _load_activity_ages_lists(self):
        self._activity_ages_lists = self._activity_ages.groupby(level=[0, 1]).agg(list)

    def _predict_activity(self, adjusted_ages):
        return pd.concat(
            {
                (h, s): pd.Series(
                    np.interp(adjusted_ages.loc[s], self._all_ages, v),
                    index=self._ages,
                )
                for (h, s), v in self._activity_ages_lists.items()
            }
        )
