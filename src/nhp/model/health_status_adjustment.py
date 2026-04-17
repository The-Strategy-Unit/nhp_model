"""Health Status Adjustment."""

from typing import List

import numpy as np
import pandas as pd
from metalog_jax.base import MetalogParameters, MetalogRandomVariableParameters
from metalog_jax.metalog import Metalog
from metalog_jax.utils import JaxUniformDistributionParameters

from nhp.model.data import Data, reference


class HealthStatusAdjustment:
    """Health Status Adjustment.

    Handles the logic for the health status adjustment in the model.
    """

    # Disability-Free Life Expectancy (DFLE) by sex
    DFLE_MALE = 10.45
    DFLE_FEMALE = 10.66
    HSA_REFERENCE_AGE: int = 65

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

        self._load_activity_ages(data_loader)
        self.base_year = int(base_year)
        # the age range that health status adjustment runs for
        # hardcoded to max out at 90 as ages >90 are mapped to 90
        self._ages = np.arange(55, 91)
        self._cache = {}

    def _load_activity_ages(self, data_loader: Data):
        self._activity_ages = (
            data_loader.get_hsa_activity_table().set_index(["hsagrp", "sex", "age"]).sort_index()
        )["activity"]

    def generate_hsa_adjusted_ages(self, end_year: int, hsa_params: dict) -> dict[int, pd.Series]:
        """Generate HSA Adjusted Ages.

        Args:
            end_year: The year the model is running for.
            hsa_params: The health status adjustment parameters.
        """
        ex_dat = reference.life_expectancy(self.base_year, end_year)

        adjusted_age = {}
        for sex in [1, 2]:
            ex_diff = ex_dat.loc[(end_year, sex)] - ex_dat.loc[(self.base_year, sex)]
            v = ex_dat.loc[(end_year, sex)].index.to_numpy() - hsa_params[sex] * ex_diff
            v.name = "adjusted_age"
            adjusted_age[sex] = v

        return pd.concat(adjusted_age, names=["sex"])

    @staticmethod
    def generate_params(
        start_year: int,
        end_year: int,
        seed: int,
        model_runs: int,
    ) -> dict[int, np.ndarray]:
        """Generate Health Status Adjustment Parameters.

        Args:
            start_year: The baseline year for the model.
            end_year: The year the model is running for.
            seed: Random seed for reproducibility.
            model_runs: Number of Model Runs.

        Returns:
            Parameters for the health status adjustment.
        """
        ex_dat = reference.life_expectancy(start_year, end_year)

        dists = reference.hsa_metalog_parameters(end_year)

        rv_params = MetalogRandomVariableParameters(
            prng_params=JaxUniformDistributionParameters(seed=seed),
            size=model_runs,
        )

        samples = {}
        for sex, dfle in [
            (1, HealthStatusAdjustment.DFLE_MALE),
            (2, HealthStatusAdjustment.DFLE_FEMALE),
        ]:
            qoi_samples = dists[sex].rvs(rv_params)

            target_ex = ex_dat[(end_year, sex, HealthStatusAdjustment.HSA_REFERENCE_AGE)]
            base_ex = ex_dat[(start_year, sex, HealthStatusAdjustment.HSA_REFERENCE_AGE)]
            numerator = qoi_samples / 100 * target_ex - dfle
            denominator = target_ex - base_ex
            # insert a 1 for the baseline model run, which is model run = 0
            samples[sex] = np.insert(np.array(numerator / denominator), 0, 1)

        return samples

    def run(self, run_params: dict) -> pd.Series:
        """Return factor for health status adjustment.

        Args:
            run_params: The run parameters.

        Returns:
            The health status adjustment factor.
        """
        hsa_param = run_params["health_status_adjustment"]
        cache_key = run_params["model_run"]
        if cache_key in self._cache:
            return self._cache[cache_key]

        adjusted_ages = self.generate_hsa_adjusted_ages(run_params["year"], hsa_param)

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
