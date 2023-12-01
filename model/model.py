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
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.health_status_adjustment import HealthStatusAdjustment
from model.helpers import age_groups, create_time_profiles, inrange, rnorm
from model.model_run import ModelRun


class Model:
    """Model

    This is a generic implementation of the model. Specific implementations of the model inherit
    from this class.

    :param model_type: the type of model, either "aae", "ip", or "op"
    :type model_type: str
    :param params: the parameters to run the model with
    :type params: dict
    :param data_path: the path to where the data files live
    :type data_path: str
    :param columns_to_load: a list of columns to load from the data
    :type columns_to_load: str, optional

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
        params: dict,
        data_path: str,
        hsa: Any,
        run_params: dict = None,
    ) -> None:
        valid_model_types = ["aae", "ip", "op"]
        assert (
            model_type in valid_model_types
        ), "Model type must be one of 'aae', 'ip', or 'op'"
        self.model_type = model_type
        #
        self.params = params
        # add model runtime if it doesn't exist
        if not "create_datetime" in self.params:
            self.params["create_datetime"] = f"{datetime.now():%Y%m%d_%H%M%S}"
        # store the path where the data is stored and the results are stored
        self._data_path = f"{data_path}/{params['start_year']}/{self.params['dataset']}"
        # load the data. we only need some of the columns for the model, so just load what we need
        self.data = self._load_parquet(self.model_type).sort_values("rn")
        self.data["age_group"] = age_groups(self.data["age"])
        self._load_strategies()
        self._load_demog_factors()
        self.hsa = hsa
        # generate the run parameters if they haven't been passed in
        self.run_params = run_params or self.generate_run_params(params)
        # get the data mask and baseline counts
        self.data_mask = self._get_data_mask()
        # pylint: disable=assignment-from-no-return
        self.baseline_counts = self._get_data_counts(self.data)

    def _load_parquet(self, file: str) -> pd.DataFrame:
        """Load a parquet file

        Helper method for loading a parquet file into a pandas DataFrame.

        You can selectively load columns by passing an array of column names to `args`.

        :param file: the name of the file to load (without the .parquet file extension)
        :type file: str
        :param args: the list of columns to load. If left blank, it loads all available columns
        :type args: [str]

        :returns: the contents of the file
        :rtype: pandas.DataFrame
        """
        return pd.read_parquet(os.path.join(self._data_path, f"{file}.parquet"))

    def _load_strategies(self) -> None:
        """Load a set of strategies"""
        # to be implemented by the concrete classes

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
        dfp = self.params["demographic_factors"]
        start_year = str(self.params["start_year"])
        years = (
            np.arange(self.params["start_year"], self.params["end_year"]) + 1
        ).astype("str")

        merge_cols = ["age", "sex"]

        demog_factors = pd.read_csv(os.path.join(self._data_path, dfp["file"]))
        demog_factors[merge_cols] = demog_factors[merge_cols].astype(int)
        demog_factors.set_index(["variant"] + merge_cols, inplace=True)

        self.demog_factors = demog_factors[years].apply(
            lambda x: x / demog_factors[start_year]
        )

    @staticmethod
    def generate_run_params(params):
        """Generate the values for each model run from the params

        Our parameters are given as intervals, this will generate a single value for each model run,
        i.e. it will return a new dictionary with an array of N+1 values (+1 for the principal model
        run).

        This method saves the values to self.run_params
        """
        # the principal projection will have run number 0. all other runs start indexing from 1
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
            # for the principal projection, just use the midpoint of the interval
            if model_run == 0:
                return sum(i) / 2
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
        return {
            "variant": [variants[np.argmax(probabilities)]]
            + rng.choice(variants, model_runs - 1, p=probabilities).tolist(),
            "seeds": rng.integers(0, 65535, model_runs).tolist(),
            "waiting_list_adjustment": params["waiting_list_adjustment"],
            "health_status_adjustment": HealthStatusAdjustment.generate_params(
                params["start_year"],
                params["end_year"],
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
                    ("expat", inrange_0_1),
                    ("repat_local", inrange_1_5),
                    ("repat_nonlocal", inrange_1_5),
                    ("baseline_adjustment", inrange_0_5),
                    ("non-demographic_adjustment", inrange_0_5),
                    ("activity_avoidance", inrange_0_1),
                    ("efficiencies", inrange_0_1),
                ]
            },
            # handle this case separately as we need to filter the params list
            "bed_occupancy": generate_param_values(
                {
                    k: v
                    for k, v in params["bed_occupancy"].items()
                    if k != "specialty_mapping"
                },
                inrange_0_1,
            ),
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

        horizon_years = self.params["end_year"] - self.params["start_year"]
        year = model_run - self.params["model_runs"]
        if year <= 0:
            year = horizon_years

        time_profiles = create_time_profiles(horizon_years, year)

        def get_param_value(prm, i):
            if isinstance(prm, dict):
                # a function to choose the next value of `i`
                #   * if `i` is a dictionary, then we need to choose the item
                #   * otherwise, `i` is already the name of a time profile, so use a const function
                get_i = (lambda k: i[k]) if isinstance(i, dict) else (lambda _: i)
                return {k: get_param_value(v, get_i(k)) for k, v in prm.items()}

            # get the time profile value, if it's a step change then we need to parse the year out
            # and run the step function
            if i[:4] == "step":
                j = int(i[4:]) - self.params["start_year"]
                i = time_profiles["step"](j)
            else:
                i = time_profiles[i]

            if isinstance(prm, list):
                return 1 - (1 - prm[model_run]) * i
            else:
                return prm * i

        time_profile_mappings = self.params["time_profile_mappings"]

        hsa = params["health_status_adjustment"][model_run]

        if model_run <= self.params["model_runs"]:
            # when we aren't performing the time profiles, set everything to "none"
            time_profile_mappings = {k: "none" for k in time_profile_mappings}
        else:
            # for the time profiles, use the principal projection
            model_run = 0

        return {
            "year": year + self.params["start_year"],
            "variant": params["variant"][model_run],
            "health_status_adjustment": hsa,
            "seed": params["seeds"][model_run],
            **{
                k: get_param_value(params[k], v)
                for k, v in time_profile_mappings.items()
            },
        }

    def _get_data_counts(self, data) -> npt.ArrayLike:
        pass

    def _get_data_mask(self) -> npt.ArrayLike:
        return np.array([1.0])

    def _convert_step_counts(self, step_counts: dict, names: list) -> dict:
        """Convert the step counts

        :param step_counts: the step counts dictionary
        :type step_counts: dict
        :param names: the names of the items in the step counts
        :type names: list
        :return: the step counts for uploading
        :rtype: dict
        """
        return {
            frozenset(
                {
                    ("activity_type", self.model_type),
                    ("change_factor", k0),
                    ("strategy", k1),
                    ("measure", k2),
                }
            ): float(v)
            for (k0, k1), vs in step_counts.items()
            for (k2, v) in zip(names, vs)
        }

    @staticmethod
    def _create_agg(model_results, cols=None, name=None, include_measure=True):
        """Create an aggregation

        Aggregates the model results across the given columns by summing the `value` column.

        It always includes the `pod` column, and by default includes the `measure` column.

        :param model_results: a DataFrame containing the results of a model run
        :type model_results: pandas.DataFrame
        :param cols: a list of column names to aggregate by
        :type cols: [str], optional
        :param name: the name to give this aggregation
        :type name: str, optional
        :param include_measure: whether to include the `measure` column
        :type include_measure: bool, optional

        :returns: a dictionary containing the aggregation
        :rtype: dict
        """
        if name is None:
            name = "+".join(cols) if cols else "default"
        cols = (
            ["sitetret", "pod"]
            + (["measure"] if include_measure else [])
            + (cols if cols else [])
        )

        agg = model_results.groupby(cols)["value"].sum()
        agg.index.names = agg.index.names[:-1] + ["measure"]

        return {name: {frozenset(zip(cols, k)): v for k, v in agg.items()}}

    # pylint: disable=invalid-name
    def go(self, model_run: int) -> dict:
        """Run the model and get the aggregated results

        Needed for running in a multiprocessing pool as you need a serializable method.

        :param model_run: the model run number we want to run
        :type model_run: _type_
        :return: the aggregated model results
        :rtype: dictionary
        """
        return ModelRun(self, model_run).get_aggregate_results()
