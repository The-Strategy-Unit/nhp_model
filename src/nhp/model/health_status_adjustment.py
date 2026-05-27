"""Health Status Adjustment."""

import numpy as np
import pandas as pd
from metalog_jax.base import MetalogRandomVariableParameters
from metalog_jax.utils import JaxUniformDistributionParameters

from nhp.model.data import Data, reference


class HealthStatusAdjustment:
    """Health Status Adjustment.

    Handles the logic for the health status adjustment in the model.
    """

    HSA_BASE_YEAR: int = 2021
    # Disability-Free Life Expectancy (DFLE) by sex
    DFLE_MALE = 10.45
    DFLE_FEMALE = 10.66
    HSA_REFERENCE_AGE: int = 65

    # load the static reference data files

    def __init__(
        self, data_loader: Data, start_year: int, end_year: int, seed: int, model_runs: int
    ):
        """Initialise HealthStatusAdjustment.

        Base class that should not be used directly, instead see HealthStatusAdjustmentGAM
        or HealthStatusAdjustmentInterpolated.

        Args:
            data_loader: The data loader.
            start_year: The baseline year for the model run.
            end_year: The end year for the model run.
            seed: Random seed for reproducibility.
            model_runs: Number of model runs.
        """
        self._start_year = start_year
        self._end_year = end_year
        self._model_runs = model_runs

        # the age range that health status adjustment runs for
        # hardcoded to max out at 90 as ages >90 are mapped to 90
        self._ages = np.arange(55, 91)
        self._all_ages = np.arange(0, 101)

        self._rv_params = MetalogRandomVariableParameters(
            prng_params=JaxUniformDistributionParameters(seed=seed),
            size=max(model_runs, 10_000),
        )

        self._load_activity_ages(data_loader)

        hsa_rebase = {
            k: v.mean(axis=0) - self._ages
            for k, v in self._generate_hsa_adjusted_ages(self.HSA_BASE_YEAR, start_year).items()
        }

        self.hsa_params = {
            k: v[0:model_runs] - hsa_rebase[k]
            for k, v in self._generate_hsa_adjusted_ages(start_year, end_year).items()
        }
        self._cache = {}

    def _load_activity_ages(self, data_loader: Data):
        self._activity_ages = (
            data_loader.get_hsa_activity_table().set_index(["hsagrp", "sex", "age"]).sort_index()
        )["activity"]

    def _generate_hsa_adjusted_ages(self, start_year: int, end_year) -> dict[int, np.ndarray]:
        """Generate HSA Adjusted Ages.

        Args:
            start_year: The baseline year for adjustment.
            end_year: The end year for adjustment.

        Returns:
            A dictionary containing the adjusted ages for each model run for each sex.
        """
        hsa_params = self._generate_params(start_year, end_year)

        ex_dat = reference.life_expectancy(start_year, end_year)

        adjusted_age = {}
        for sex in [1, 2]:
            h = hsa_params[sex].reshape(-1, 1)
            ex_diff = ex_dat.loc[(end_year, sex)] - ex_dat.loc[(start_year, sex)]
            adjusted_age[sex] = self._ages - h * ex_diff.to_numpy()

        return adjusted_age

    def _generate_params(self, start_year: int, end_year: int) -> dict[int, np.ndarray]:
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

        samples = {}
        for sex, dfle in [
            (1, HealthStatusAdjustment.DFLE_MALE),
            (2, HealthStatusAdjustment.DFLE_FEMALE),
        ]:
            qoi_samples = dists[sex].rvs(self._rv_params)

            target_ex = ex_dat[(end_year, sex, HealthStatusAdjustment.HSA_REFERENCE_AGE)]
            base_ex = ex_dat[(start_year, sex, HealthStatusAdjustment.HSA_REFERENCE_AGE)]
            numerator = qoi_samples / 100 * target_ex - dfle
            denominator = target_ex - base_ex
            # insert a 1 for the baseline model run, which is model run = 0
            samples[sex] = np.array(numerator / denominator)

        return samples

    def run(self, run_params: dict) -> pd.Series:
        """Return factor for health status adjustment.

        Args:
            run_params: The run parameters.

        Returns:
            The health status adjustment factor.
        """
        # model runs start from to params["model_run"], but the parameters for HSA are 0 based, so
        # we need to subtract 1.
        cache_key = run_params["model_run"] - 1
        if cache_key in self._cache:
            return self._cache[cache_key]

        adjusted_ages = pd.concat(
            {k: pd.Series(v[cache_key], index=self._ages) for k, v in self.hsa_params.items()}
        )
        adjusted_ages.index.names = ["sex", "age"]

        factor = (
            self._predict_activity(adjusted_ages).rename_axis(["hsagrp", "sex", "age"])
            / self._activity_ages.loc[slice(None), slice(None), self._ages]
        ).rename("health_status_adjustment")

        # if any factor goes below 0, set it to 0
        factor[factor < 0] = 0

        self._cache[cache_key] = factor
        return factor

    def _predict_activity(self, adjusted_ages):
        raise NotImplementedError()


class HealthStatusAdjustmentGAM(HealthStatusAdjustment):
    """Health Status Adjustment (GAMs)."""

    def __init__(
        self, data_loader: Data, start_year: int, end_year: int, seed: int, model_runs: int
    ):
        """Initialise HealthStatusAdjustmentGAM.

        Args:
            data_loader: The data loader.
            start_year: The baseline year for the model run.
            end_year: The end year for the model run.
            seed: Random seed for reproducibility.
            model_runs: Number of model runs.
        """
        self._gams = data_loader.get_hsa_gams()

        super().__init__(data_loader, start_year, end_year, seed, model_runs)

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

    def __init__(
        self, data_loader: Data, start_year: int, end_year: int, seed: int, model_runs: int
    ):
        """Initialise HealthStatusAdjustmentInterpolated.

        Args:
            data_loader: The data loader.
            start_year: The baseline year for the model run.
            end_year: The end year for the model run.
            seed: Random seed for reproducibility.
            model_runs: Number of model runs.
        """
        super().__init__(data_loader, start_year, end_year, seed, model_runs)
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
                for (h, s), v in self._activity_ages_lists.items()
            }
        )
