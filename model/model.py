"""Model Module

Implements the generic class for all other model types. This should not be directly instantiated,
instead you should use one of the concrete classes (e.g. :class:`model.aae.AaEModel`,
:class:`model.inpatients.InpatientsModel`, :class:`model.outpatients.OutpatientsModel`).

You can then run a model using the `run` method, and aggregate the results using the `aggregate`
method.
"""

import os
from datetime import datetime
from functools import partial
from typing import Any, Callable, List

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.data import Data
from model.health_status_adjustment import (
    HealthStatusAdjustment,
    HealthStatusAdjustmentInterpolated,
)
from model.helpers import age_groups, inrange, load_params, rnorm
from model.model_run import ModelRun


class Model:
    """Model

    This is a generic implementation of the model. Specific implementations of the model inherit
    from this class.

    :param model_type: the type of model, either "aae", "ip", or "op"
    :type model_type: str
    :param measures: the names of the measures in the model
    :type measures: str
    :param params: the parameters to run the model with, or the path to a params file to load
    :type params: dict or string
    :param data: a Data class ready to be constructed
    :type data: Data
    :param hsa: An instance of the HealthStatusAdjustment class. If left as None an instance is
    created
    :type hsa: HealthStatusAdjustment, optional
    :param run_params: the parameters to use for each model run. generated automatically if left as
    None
    :type run_params: dict
    :param save_full_model_results: whether to save the full model results or not
    :type save_full_model_results: bool, optional

    :ivar str model_type: a string describing the type of model, must be one of "aae", "ip", or "op"
    :ivar dict params: the parameters that are to be used by this model instance
    :ivar pandas.DataFrame data: the HES data extract used by this model instance

    :Example:

    >>> from model.aae import AaEModel
    >>> from model.helpers import load_params
    >>> params = load_params("sample_params.json")
    >>> m = AaAModel(params, "data")
    >>> change_factors, results = m.run(0)
    >>> aggregated_results = m.aggregate(reuslts, 0)
    """

    def __init__(
        self,
        model_type: str,
        measures: str,
        params: dict,
        data: Data,
        hsa: Any = None,
        run_params: dict = None,
        save_full_model_results: bool = False,
    ) -> None:
        valid_model_types = ["aae", "ip", "op"]
        assert (
            model_type in valid_model_types
        ), "Model type must be one of 'aae', 'ip', or 'op'"
        self.model_type = model_type
        #
        if isinstance(params, str):
            params = load_params(params)
        self.params = params
        # add model runtime if it doesn't exist
        if not "create_datetime" in self.params:
            self.params["create_datetime"] = f"{datetime.now():%Y%m%d_%H%M%S}"
        self._data_loader = data(params["start_year"], params["dataset"])
        #
        self._measures = measures
        # load the data. we only need some of the columns for the model, so just load what we need
        self._load_data()
        self._load_strategies()
        self._load_demog_factors()
        # create HSA object if it hasn't been passed in
        year = params["start_year"]
        self.hsa = hsa or HealthStatusAdjustmentInterpolated(
            self._data_loader, str(year)
        )
        # generate the run parameters if they haven't been passed in
        self.run_params = run_params or self.generate_run_params(params)
        #
        self.save_full_model_results = save_full_model_results
        #
        self._data_loader = None

    def _add_pod_to_data(self) -> None:
        """Adds the POD column to data"""
        # to be implemented in ip/op/aae

    @property
    def measures(self) -> List[str]:
        """The names of the measure columns

        :return: the names of the measure columns
        :rtype: List[str]
        """
        return self._measures

    def _get_data(self) -> None:
        """Load the data"""
        # to be implemented by the concrete classes

    def _load_data(self) -> None:
        self.data = self._get_data().sort_values("rn")
        self.data["age_group"] = age_groups(self.data["age"])
        # pylint: disable=assignment-from-no-return
        self.baseline_counts = self.get_data_counts(self.data)
        self._add_pod_to_data()

        self.baseline_step_counts = (
            pd.DataFrame(
                self.baseline_counts.transpose(),
                columns=self.measures,
                index=pd.MultiIndex.from_frame(self.data[["pod", "sitetret"]]),
            )
            .groupby(level=[0, 1])
            .sum()
            .reset_index()
            .assign(change_factor="baseline", strategy="-")
        )

    def _load_strategies(self) -> None:
        """Load a set of strategies"""
        # to be implemented by the concrete classes
        self.strategies = None  # lint helper

    def _load_demog_factors(self) -> None:
        """Load the demographic factors

        Load the demographic factors csv file and calculate the demographics growth factor for the
        years in the parameters.

        Creates 3 private variables:

          * | `self.demog_factors`: a pandas.DataFrame which has a 1:1 correspondance to the model
            | data and a column for each of the different population projections available
          * | `self._variants`: a list containing the names of the different population projections
            | available
          * `self._probabilities`: a list containing the probability of selecting a given variant
        """
        start_year = str(self.params["start_year"])
        years = (
            np.arange(self.params["start_year"], self.params["end_year"]) + 1
        ).astype("str")

        merge_cols = ["age", "sex"]

        def load_factors(factors):
            factors[merge_cols] = factors[merge_cols].astype(int)
            factors.set_index(["variant"] + merge_cols, inplace=True)

            return factors[years].apply(lambda x: x / factors[start_year])

        self.demog_factors = load_factors(self._data_loader.get_demographic_factors())
        self.birth_factors = load_factors(self._data_loader.get_birth_factors())

    @staticmethod
    def generate_run_params(params):
        """Generate the values for each model run from the params

        Our parameters are given as intervals, this will generate a single value for each model run,
        i.e. it will return a new dictionary with an array of N+1 values (+1 for the principal model
        run).

        This method saves the values to self.run_params
        """
        # the baseline run will have run number 0. all other runs start indexing from 1
        rng = np.random.default_rng(params["seed"])
        model_runs = params["model_runs"] + 1  # add 1 for the principal

        # partially apply inrange to give us the different type of ranges we use
        inrange_0_1 = partial(inrange, low=0, high=1)
        inrange_0_5 = partial(inrange, low=0, high=5)
        inrange_1_5 = partial(inrange, low=1, high=5)

        # function to generate a value for a model run
        def gen_value(model_run, i):
            # we haven't received an interval, just a single value
            if not isinstance(i, list):
                return i
            # for the baseline run, no params need to be set
            if model_run == 0:
                return 1
            return rnorm(rng, *i)  # otherwise

        # function to traverse a dictionary until we find the values to generate from
        def generate_param_values(prm, valid_range):
            if isinstance(prm, dict):
                if "interval" not in prm:
                    return {
                        k: generate_param_values(v, valid_range) for k, v in prm.items()
                    }
                prm = prm["interval"]
            return [valid_range(gen_value(mr, prm)) for mr in range(model_runs)]

        variants = list(params["demographic_factors"]["variant_probabilities"].keys())
        probabilities = list(
            params["demographic_factors"]["variant_probabilities"].values()
        )
        # there can be numerical inacurracies in the probabilities not summing to 1
        # this will adjust the probabiliteies to sum correctly
        probabilities = [p / sum(probabilities) for p in probabilities]

        variants = [variants[np.argmax(probabilities)]] + rng.choice(
            variants, model_runs - 1, p=probabilities
        ).tolist()

        return {
            "variant": variants,
            "seeds": rng.integers(0, 65535, model_runs).tolist(),
            "health_status_adjustment": HealthStatusAdjustment.generate_params(
                params["start_year"],
                params["end_year"],
                variants,
                rng,
                model_runs - 1,
            ),
            # generate param values for the different items in params: this will traverse the dicts
            # until a value is reached that isn't a dict. Then it will generate the required amount
            # of values for that parameter
            **{
                k: generate_param_values(params[k], v)
                for k, v in [
                    ("covid_adjustment", lambda x: x),
                    ("waiting_list_adjustment", inrange_0_5),
                    ("expat", inrange_0_1),
                    ("repat_local", inrange_1_5),
                    ("repat_nonlocal", inrange_1_5),
                    ("baseline_adjustment", inrange_0_5),
                    ("non-demographic_adjustment", inrange_0_5),
                    ("activity_avoidance", inrange_0_1),
                    ("efficiencies", inrange_0_1),
                ]
            },
        }

    def _get_run_params(self, model_run: int) -> dict:
        """Gets the parameters for a particular model run

        Takes the run_params and extracts a single item from the arrays for the current model run

        :param model_run: the current model run
        :type model_run: int

        :returns: the run parameters for the given model run
        :rtype: dict
        """
        params = self.run_params

        def get_param_value(prm):
            if isinstance(prm, dict):
                return {k: get_param_value(v) for k, v in prm.items()}

            return prm[model_run]

        return {
            "year": self.params["end_year"],
            "seed": params["seeds"][model_run],
            **{k: get_param_value(params[k]) for k in params.keys() if k != "seeds"},
        }

    def get_data_counts(self, data: pd.DataFrame) -> npt.ArrayLike:
        """Get row counts of data

        :param data: the data to get the counts of
        :type data: pd.DataFrame
        :return: the counts of the data, required for activity avoidance steps
        :rtype: npt.ArrayLike
        """
        raise NotImplementedError()

    def activity_avoidance(self, data: pd.DataFrame, model_run: ModelRun) -> dict:
        """perform the activity avoidance (strategies)"""
        # if there are no items in params for activity_avoidance then exit
        if not (params := model_run.run_params["activity_avoidance"][self.model_type]):
            return data, None

        rng = model_run.rng

        strategies = self.strategies["activity_avoidance"]
        # decide whether to sample a strategy for each row this model run
        strategies = strategies.loc[
            rng.binomial(1, strategies["sample_rate"]).astype("bool"), "strategy"
        ]
        # join the parameters, then pivot wider
        strategies = (
            strategies.reset_index()
            .merge(pd.Series(params, name="aaf"), left_on="strategy", right_index=True)
            .pivot(index="rn", columns="strategy", values="aaf")
        )

        data_counts = self.get_data_counts(data)

        factors_aa = (
            data[["rn"]]
            .merge(strategies, left_on="rn", right_index=True, how="left")
            .fillna(1)
            .drop(columns="rn")
        )

        row_samples = rng.binomial(data_counts.astype("int"), factors_aa.prod(axis=1))

        step_counts = (
            model_run.fix_step_counts(
                data,
                row_samples,
                factors_aa,
                "activity_avoidance_interaction_term",
            )
            .rename(columns={"change_factor": "strategy"})
            .assign(change_factor="activity_avoidance")
        )

        return (
            self.apply_resampling(row_samples, data),
            step_counts,
        )

    def calculate_avoided_activity(
        self, data: pd.DataFrame, data_resampled: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate the rows that have been avoided

        :param data: The data before the binomial thinning step
        :type data: pd.DataFrame
        :return: The data that was avoided in the binomial thinning step
        :rtype: pd.DataFrame
        """
        # implemented by concrete classes

    # pylint: disable=invalid-name
    def go(self, model_run: int) -> dict:
        """Run the model and get the aggregated results

        Needed for running in a multiprocessing pool as you need a serializable method.

        :param model_run: the model run number we want to run
        :type model_run: int
        :return: the aggregated model results
        :rtype: dictionary
        """
        mr = ModelRun(self, model_run)

        if self.save_full_model_results and model_run > 0:
            dataset = self.params["dataset"]
            scenario = self.params["scenario"]
            create_datetime = self.params["create_datetime"]

            def path_fn(f):
                path = f"results/{dataset}/{scenario}/{create_datetime}/{f}/model_run={model_run}/"
                os.makedirs(path, exist_ok=True)
                return path

            self.save_results(mr, path_fn)

        return mr.get_aggregate_results()

    @staticmethod
    def get_agg(results: pd.DataFrame, *args) -> pd.Series:
        """Get aggregation from model results

        :param results: The results of a model run
        :type results: pd.DataFrame
        :param *args: Any columns you wish to aggregate on
        :return: Aggregated results
        :rtype: pd.Series
        """
        return results.groupby(["pod", "sitetret", *args, "measure"])["value"].sum()

    def save_results(self, model_run: ModelRun, path_fn: Callable[[str], str]) -> None:
        """Save the results of running the model

        This method is used for saving the results of the model run to disk as a parquet file.
        It saves just the `rn` (row number) column and the `arrivals`, with the intention that
        you rejoin to the original data.

        :param model_results: a DataFrame containing the results of a model iteration
        :param path_fn: a function which takes the activity type and returns a path
        """
        # implemented by concrete classes
