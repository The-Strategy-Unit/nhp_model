"""_summary_"""
# pylint: disable=too-few-public-methods

import numpy as np
import pandas as pd


class HealthStatusAdjustment:
    """Health Status Adjustment

    handles the logic for the health status adjustment in the model"""

    def __init__(self, data_path: str, life_expectancy: dict):
        self._all_ages = np.arange(0, 101)
        self._activity_table_path = f"{data_path}/hsa_activity_table.csv"

        self._load_life_expectancy_series(life_expectancy)
        self._load_activity_ages()
        self._cache = {}

    def _load_life_expectancy_series(self, life_expectancy):
        lep = life_expectancy
        self._ages = np.arange(lep["min_age"], lep["max_age"] + 1)

        self._life_expectancy = pd.concat(
            {
                si: pd.Series(lep[ss], index=self._ages, name="life_expectancy")
                for (si, ss) in [(1, "m"), (2, "f")]
            }
        ).rename_axis(["sex", "age"])

    def _load_activity_ages(self):
        self._activity_ages = (
            pd.read_csv(self._activity_table_path).set_index(["hsagrp", "sex", "age"])
        )["activity"]

    def run(self, hsa_param: float):
        """_summary_

        :param hsa_param: _description_
        :type hsa_param: int
        :return: _description_
        :rtype: _type_
        """
        if hsa_param in self._cache:
            return self._cache[hsa_param]

        ages = self._life_expectancy.index.get_level_values(1)
        adjusted_ages = ages - self._life_expectancy * hsa_param

        self._cache[hsa_param] = factor = (
            self._predict_activity(adjusted_ages).rename_axis(["hsagrp", "sex", "age"])
            / self._activity_ages.loc[slice(None), slice(None), self._ages]
        ).rename("health_status_adjustment")

        return factor

    def _predict_activity(self, adjusted_ages):
        """"""


class HealthStatusAdjustmentGAM(HealthStatusAdjustment):
    """_summary_"""

    def __init__(self, data_path: str, life_expectancy: dict):
        import pickle  # pylint: disable=import-outside-toplevel

        with open(f"{data_path}/hsa_gams.pkl", "rb") as hsa_pkl:
            self._gams = pickle.load(hsa_pkl)

        super().__init__(data_path, life_expectancy)

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

    def __init__(self, data_path: str, life_expectancy: dict):
        super().__init__(data_path, life_expectancy)
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
