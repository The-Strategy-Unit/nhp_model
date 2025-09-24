"""NHP Data Loaders.

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

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

        :return: the inpatients dataframe
        :rtype: pd.DataFrame
        """
        raise NotImplementedError()

    def get_ip_strategies(self) -> dict[str, pd.DataFrame]:
        """Get the inpatients strategies dataframe.

        :return: the inpatients strategies dataframe
        :rtype: pd.DataFrame
        """
        raise NotImplementedError()

    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe.

        :return: the outpatients dataframe
        :rtype: pd.DataFrame
        """
        raise NotImplementedError()

    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe.

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        raise NotImplementedError()

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe.

        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """
        raise NotImplementedError()

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        raise NotImplementedError()

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        raise NotImplementedError()

    def get_hsa_gams(self):
        """Get the health status adjustment gams."""
        raise NotImplementedError()

    def get_inequalities(self):
        """Get the inequalities dataframe.

        :return: the inequalities dataframe
        :rtype: pd.DataFrame
        """
        raise NotImplementedError()
