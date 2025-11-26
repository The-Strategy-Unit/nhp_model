"""Health Status Adjustment."""

from math import pi, sqrt
from typing import List

import numpy as np
import pandas as pd

from nhp.model.data import Data, reference


class HealthStatusAdjustment:
    """Health Status Adjustment.

    Handles the logic for the health status adjustment in the model.
    """

    # load the static reference data files

    def __init__(self, data_loader: Data, base_year: str):
        """Initialise HealthStatusAdjustment.

        Base class that should not be used directly, instead see HealthStatusAdjustmentGAM
        or HealthStatusAdjustmentInterpolated.

        Args:
            data_loader: The data loader.
            base_year: The baseline year for the model run.
        """
        self._all_ages = np.arange(0, 101)

        self._load_life_expectancy_series(base_year)
        self._load_activity_ages(data_loader)
        self._cache = {}

    def _load_life_expectancy_series(self, base_year: str):
        # the age range that health status adjustment runs for
        # hardcoded to max out at 90 as ages >90 are mapped to 90
        self._ages = np.arange(55, 91)
        # load the life expectancy file, only select the rows for the ages we are interested in
        lexc = reference.life_expectancy().set_index(["var", "sex", "age"])
        lexc = lexc[lexc.index.isin(self._ages, level=2)]
        # calculate the life expectancy (change) between the model year and base year
        self._life_expectancy = lexc.apply(lambda x: x - lexc[str(base_year)])

    def _load_activity_ages(self, data_loader: Data):
        self._activity_ages = (
            data_loader.get_hsa_activity_table().set_index(["hsagrp", "sex", "age"]).sort_index()
        )["activity"]

    @staticmethod
    def generate_params(
        start_year: int,
        end_year: int,
        variants: List[str],
        rng: np.random.Generator,
        model_runs: int,
    ) -> np.ndarray:
        """Generate Health Status Adjustment Parameters.

        Args:
            start_year: The baseline year for the model.
            end_year: The year the model is running for.
            variants: List of population variants.
            rng: Random Number Generator.
            model_runs: Number of Model Runs.

        Returns:
            Parameters for the health status adjustment.
        """
        hsa_snp = reference.split_normal_params().set_index(["var", "sex", "year"])

        def gen(variant, sex):
            mode: float
            sd1: float
            sd2: float
            mode, sd1, sd2 = hsa_snp.loc[(variant, sex, end_year)]  # type: ignore

            return np.concatenate(
                [
                    [mode],
                    HealthStatusAdjustment.random_splitnorm(rng, model_runs, mode, sd1, sd2),
                    hsa_snp.loc[(variant, sex, np.arange(start_year + 1, end_year)), "mode"],  # type: ignore
                ]
            )

        values = {
            v: np.transpose([gen(v, "m"), gen(v, "f")])
            for v in hsa_snp.index.levels[0]  # type: ignore
        }

        variant_lookup = reference.variant_lookup()
        return np.array(
            [
                values[variant_lookup[v]][i]
                for i, v in enumerate(variants + variants[0:1] * (end_year - start_year - 1))
            ]
        )

    @staticmethod
    def random_splitnorm(
        rng: np.random.Generator,
        n: int,
        mode: float,
        sd1: float,
        sd2: float,
    ) -> np.ndarray:
        """Generate random splitnormal values.

        Args:
            rng: Random Number Generator.
            n: Number of random values to generate.
            mode: The mode of the distribution.
            sd1: The standard deviation of the left side of the distribution.
            sd2: The standard deviation of the right side of the distribution.

        Returns:
            n random number values sampled from the split normal distribution.
        """
        # lazy import for performance
        import scipy.stats as spt  # noqa: PLC0415

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

    def run(self, run_params: dict) -> pd.Series:
        """Return factor for health status adjustment.

        Args:
            run_params: The run parameters.

        Returns:
            The health status adjustment factor.
        """
        hsa_param = run_params["health_status_adjustment"]
        selected_variant = reference.variant_lookup()[run_params["variant"]]
        cache_key = (*hsa_param, selected_variant)
        if cache_key in self._cache:
            return self._cache[cache_key]

        lexc = self._life_expectancy.loc[(selected_variant, slice(None), slice(None))][  # type: ignore
            str(run_params["year"])
        ]
        hsa_param = np.repeat(hsa_param, len(self._ages))
        adjusted_ages = np.tile(self._ages, 2) - lexc * hsa_param

        factor = (
            self._predict_activity(adjusted_ages).rename_axis(["hsagrp", "sex", "age"])
            / self._activity_ages.loc[slice(None), slice(None), self._ages]  # type: ignore
        ).rename("health_status_adjustment")

        # if any factor goes below 0, set it to 0
        factor[factor < 0] = 0

        self._cache[cache_key] = factor
        return factor

    def _predict_activity(self, adjusted_ages):
        raise NotImplementedError()


class HealthStatusAdjustmentGAM(HealthStatusAdjustment):
    """Health Status Adjustment (GAMs)."""

    def __init__(self, data: Data, base_year: str):
        """Initialise HealthStatusAdjustmentGAM.

        Args:
            data: The data loader.
            base_year: The baseline year for the model run.
        """
        self._gams = data.get_hsa_gams()

        super().__init__(data, base_year)

    def _predict_activity(self, adjusted_ages):
        return pd.concat(
            {
                (h, s): pd.Series(
                    g.predict(adjusted_ages.loc[s]),
                    index=self._ages,
                ).apply(lambda x: x if x > 0 else 0)
                for (h, s), g in self._gams.items()
            }
        )


class HealthStatusAdjustmentInterpolated(HealthStatusAdjustment):
    """Health Status Adjustment (Interpolated)."""

    def __init__(self, data: Data, base_year: str):
        """Initialise HealthStatusAdjustmentInterpolated.

        Args:
            data: The data loader.
            base_year: The baseline year for the model run.
        """
        super().__init__(data, base_year)
        self._load_activity_ages_lists()

    def _load_activity_ages_lists(self):
        self._activity_ages_lists = self._activity_ages.groupby(level=[0, 1]).agg(list)

    def _predict_activity(self, adjusted_ages):
        return pd.concat(
            {
                (h, s): pd.Series(
                    np.interp(adjusted_ages.loc[s], self._all_ages, v),
                    index=self._ages,
                ).apply(lambda x: x if x > 0 else 0)
                for (h, s), v in self._activity_ages_lists.items()  # type: ignore
            }
        )
