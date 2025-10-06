"""NHP Data Loaders.

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

from typing import Any

import pandas as pd


class Data:
    """Load NHP data.

    Interface for loading data for the NHP model. This interface should have no concrete
    implementations, instead other classes should derive from this interface.
    """

    def __init__(self):
        """Initialise Data data loader class."""
        pass

    def get_ip(self) -> pd.DataFrame:
        """Get the inpatients dataframe.

        Returns:
            The inpatients dataframe.
        """
        raise NotImplementedError()

    def get_ip_strategies(self) -> dict[str, pd.DataFrame]:
        """Get the inpatients strategies dataframe.

        Returns:
            The inpatients strategies dataframe.
        """
        raise NotImplementedError()

    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe.

        Returns:
            The outpatients dataframe.
        """
        raise NotImplementedError()

    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe.

        Returns:
            The A&E dataframe.
        """
        raise NotImplementedError()

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe.

        Returns:
            The birth factors dataframe.
        """
        raise NotImplementedError()

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        Returns:
            The demographic factors dataframe.
        """
        raise NotImplementedError()

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        Returns:
            The demographic factors dataframe.
        """
        raise NotImplementedError()

    def get_hsa_gams(self) -> Any:
        """Get the health status adjustment gams.

        Returns:
            The health status adjustment gams.
        """
        raise NotImplementedError()

    def get_inequalities(self) -> pd.DataFrame:
        """Get the inequalities dataframe.

        Returns:
            The inequalities dataframe.
        """
        raise NotImplementedError()
