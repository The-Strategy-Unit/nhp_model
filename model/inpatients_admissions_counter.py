"""Inpatients Admissions Counter"""

import numpy as np
import pandas as pd


class InpatientsAdmissionsCounter:
    """Inpatients Admissions Counter

    Keeps track of how many times each row of data should be sampled.

    :param data: the data that we are going to sample from
    :type data: pandas.DataFrame
    :param rng: a random number generator created for each model iteration
    :type rng: numpy.random.Generator

    :ivar dict step_counts: the change to the admissions and beddays caused by each step
    """

    def __init__(self, data, rng):
        self._data = data
        self._rng = rng
        self._beddays = (data["speldur"] + 1).astype(int)
        self._admissions = np.ones_like(data.index)
        self.step_counts = {
            ("baseline", "-"): {
                "admissions": len(self._admissions),
                "beddays": sum(self._beddays),
            }
        }

    def _update_step_counts(self, new_admissions, name, split_by=None):
        """Update step counts

        Keep track of how much each step has affected the admissions and beddays by

        :param new_admissions: the new admissions values
        :type new_admissions: numpy.ndarray
        :param name: the name of the step to insert into the step counts
        :type name: str
        :param split_by: a list of values to group the rows by
        :type split_by: list, optional

        :returns: the admissions counter object
        :rtype: InpatientsAdmissionsCounter
        """
        diff = new_admissions - self._admissions

        if split_by == None:
            self.step_counts[(name, "-")] = {
                "admissions": sum(diff),
                "beddays": sum(diff * self._beddays),
            }
        else:
            counts = (
                pd.DataFrame({"admissions": diff, "beddays": diff * self._beddays})
                .groupby(split_by)
                .sum()
                .to_dict(orient="index")
            )

            self.step_counts.update({(name, k): v for k, v in counts.items()})

        self._admissions = new_admissions
        return self

    def poisson_step(self, factor, name, split_by=None):
        """update the row counts using a poisson distribution

        :param factor: a list the length of the data for the lambda argument for the poisson distribution
        :type factor: list
        :param name: the name of the step to insert into the step counts
        :type name: str
        :param split_by: a list of values to group the rows by
        :type split_by: list, optional

        :returns: the admissions counter object
        :rtype: InpatientsAdmissionsCounter
        """
        return self._update_step_counts(
            self._rng.poisson(factor * self._admissions), name, split_by
        )

    def binomial_step(self, factor, name, split_by=None):
        """update the row counts using a binomial distribution

        :param factor: a list the length of the data for the p argument for the poisson distribution
        :type factor: list
        :param name: the name of the step to insert into the step counts
        :type name: str
        :param split_by: a list of values to group the rows by
        :type split_by: list, optional

        :returns: the admissions counter object
        :rtype: InpatientsAdmissionsCounter
        """
        return self._update_step_counts(
            self._rng.binomial(n=self._admissions, p=factor), name, split_by
        )

    def get_data(self):
        """Get the data

        :returns: the data, resampling the rows based on the all of the steps that have been performed
        :rtype: pandas.DataFrame
        """
        return self._data.loc[self._data.index.repeat(self._admissions)].reset_index(
            drop=True
        )
