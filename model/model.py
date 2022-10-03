"""Model Module

Implements the generic class for all other model types. This should not be directly instantiated,
instead you should use one of the concrete classes (e.g. :class:`model.aae.AaEModel`,
:class:`model.inpatients.InpatientsModel`, :class:`model.outpatients.OutpatientsModel`).

You can then run a model using the `run` method, and aggregate the results using the `aggregate`
method.
"""

import os
import pickle
from collections import namedtuple
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from model.helpers import age_groups, inrange, rnorm


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
        columns_to_load: list[str] = None,
    ) -> None:
        assert model_type in [
            "aae",
            "ip",
            "op",
        ], "Model type must be one of 'aae', 'ip', or 'op'"
        self.model_type = model_type
        #
        self.params = params
        # add model runtime if it doesn't exist
        if not "create_datetime" in self.params:
            self.params["create_datetime"] = f"{datetime.now():%Y%m%d_%H%M%S}"
        # store the path where the data is stored and the results are stored
        self._data_path = os.path.join(data_path, self.params["dataset"])
        # load the data that's shared across different model types
        with open(f"{self._data_path}/hsa_gams.pkl", "rb") as hsa_pkl:
            self._hsa_gams = pickle.load(hsa_pkl)
        # load the data. we only need some of the columns for the model, so just load what we need
        self.data = self._load_parquet(self.model_type, columns_to_load)
        self.data["age_group"] = age_groups(self.data["age"])
        # now data is loaded we can load the demographic factors
        self._load_demog_factors()
        # generate the run parameters
        self._generate_run_params()

    def _load_demog_factors(self) -> None:
        """Load the demographic factors

        Load the demographic factors csv file and calculate the demographics growth factor for the
        years in the parameters.

        Creates 3 private variables:

          * | `self._demog_factors`: a pandas.DataFrame which has a 1:1 correspondance to the model
            | data and a column for each of the different population projections available
          * | `self._variants`: a list containing the names of the different population projections
            | available
          * `self._probabilities`: a list containing the probability of selecting a given variant
        """
        dfp = self.params["demographic_factors"]
        start_year = str(self.params["start_year"])
        end_year = str(self.params["end_year"])

        merge_cols = ["age", "sex"]

        demog_factors = pd.read_csv(os.path.join(self._data_path, dfp["file"]))
        demog_factors[merge_cols] = demog_factors[merge_cols].astype(int)
        demog_factors["factor"] = demog_factors[end_year] / demog_factors[start_year]
        demog_factors.set_index(merge_cols, inplace=True)

        self._demog_factors = (
            self.data[["rn"] + merge_cols]
            .merge(
                demog_factors.pivot(None, "variant", "factor"),
                left_on=merge_cols,
                right_index=True,
            )
            .drop(merge_cols, axis="columns")
            .set_index("rn")
        )

        self._variants = list(dfp["variant_probabilities"].keys())
        self._probabilities = list(dfp["variant_probabilities"].values())

    def _health_status_adjustment(
        self, data: pd.DataFrame, run_params: dict
    ) -> np.ndarray:
        """Get the health status adjustment factor for the current model run

        This starts off by taking the `life_expectancy` values from the params, as well as the
        `health_status_adjustment` value for the current model run. We use these values to work
        out an "adjusted age" for our population.

        We then use the HSA GAM's (g) to work out our factor with

        .. math:: \\frac{g(age_{\\text{adjusted}})}{g(age_{\\text{actual}})}

        :param data: the data of the current model run
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict

        :returns: a list containing the HSA factor for each row in `data`
        :rtype: numpy.ndarray
        """
        # convert the arrays from the life expectancy paramerters into a data frame
        lep = self.params["life_expectancy"]
        ages = np.tile(np.arange(lep["min_age"], lep["max_age"] + 1), 2)
        sexs = np.repeat([1, 2], len(ages) // 2)
        lex = pd.DataFrame({"age": ages, "sex": sexs, "ex": lep["m"] + lep["f"]})
        # adjust the life expectancy column using the health status adjustment parameter
        lex["ex"] *= run_params["health_status_adjustment"]
        # caclulate the adjusted age
        lex["adjusted_age"] = lex["age"] - lex["ex"]

        lex.set_index("sex", inplace=True)

        # use the hsa gam's to predict with the adjusted age and the actual age to calculate the
        # health status adjustment factor
        hsa = pd.concat(
            [
                pd.DataFrame(
                    {
                        "hsagrp": h,
                        "sex": int(s),
                        "age": lex.loc[int(s), "age"],
                        "hsa_f": g.predict(lex.loc[int(s), "adjusted_age"])
                        / g.predict(lex.loc[int(s), "age"]),
                    }
                )
                for (h, s), g in self._hsa_gams.items()
            ]
        ).reset_index(drop=True)
        return (
            (
                data.reset_index()
                .merge(hsa, on=["hsagrp", "sex", "age"], how="left")
                .set_index(["rn"])
            )["hsa_f"]
            .fillna(1)
            .to_numpy()
        )

    def _load_parquet(self, file: str, *args: list[str]) -> pd.DataFrame:
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
        return pq.read_pandas(
            os.path.join(self._data_path, f"{file}.parquet"), *args
        ).to_pandas()

    def _factor_helper(
        self, data: pd.DataFrame, factor_run_params: dict, column_values: dict
    ) -> np.ndarray:
        """Get a factor column

        Helper method (for aae/op) to generate a factor column. This is intended to be used with
        columns which are binary flags.

        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param factor_run_params: a set of parameters to use for this factor, the keys of which
            will join to `data` on the `hsagrp` column
        :type factor_run_params: dict
        :param column_values: a dictionary containing a mapping between column and value, this is
            used to "filter" the `data` to only apply this factor to relevant rows

        :returns: a numpy array of the factors for each row of data
        :rtype: numpy.ndarray
        """
        hsagrps = (
            data.merge(pd.DataFrame(column_values, index=[0])).hsagrp.unique().tolist()
        )
        factor = {
            h: v
            for k, v in factor_run_params.items()
            for h in hsagrps
            if h.startswith(f"{self.model_type}_{k}")
        }

        factor = pd.DataFrame(
            {"hsagrp": factor.keys(), **column_values, "f": factor.values()}
        )

        return (
            data.merge(factor, how="left", on=list(factor.columns[:-1]))
            .f.fillna(1)
            .to_numpy()
        )

    def _generate_run_params(self):
        """Generate the values for each model run from the params

        Our parameters are given as intervals, this will generate a single value for each model run,
        i.e. it will return a new dictionary with an array of N+1 values (+1 for the principal model
        run).

        This method saves the values to self.run_params
        """
        params = self.params

        # the principal projection will have run number 0. all other runs start indexing from 1
        rng = np.random.default_rng(params["seed"])
        model_runs = params["model_runs"] + 1  # add 1 for the principal

        # partially apply inrange to give us the different type of ranges we use
        inrange_0_1 = partial(inrange, low=0, high=1)
        inrange_0_5 = partial(inrange, low=0, high=5)

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
        def generate_param_values(p, valid_range):
            if isinstance(p, dict):
                if "interval" not in p:
                    return {
                        k: generate_param_values(v, valid_range) for k, v in p.items()
                    }
                p = p["interval"]
            return [valid_range(gen_value(mr, p)) for mr in range(model_runs)]

        self.run_params = {
            "variant": [self._variants[np.argmax(self._probabilities)]]
            + rng.choice(
                self._variants, model_runs - 1, p=self._probabilities
            ).tolist(),
            "seeds": rng.integers(0, 65535, model_runs).tolist(),
            "waiting_list_adjustment": params["waiting_list_adjustment"],
            # generate param values for the different items in params: this will traverse the dicts
            # until a value is reached that isn't a dict. Then it will generate the required amount
            # of values for that parameter
            **{
                k: generate_param_values(params[k], v)
                for k, v in [
                    ("health_status_adjustment", inrange_0_5),
                    ("expat", inrange_0_1),
                    ("repat_local", inrange_0_5),
                    ("repat_nonlocal", inrange_0_5),
                    ("non-demographic_adjustment", inrange_0_5),
                    ("inpatient_factors", inrange_0_1),
                    ("outpatient_factors", inrange_0_1),
                    ("aae_factors", inrange_0_1),
                    ("theatres", inrange_0_5),
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

        def get_param_value(p):
            if isinstance(p, dict):
                return {k: get_param_value(v) for k, v in p.items()}
            return p[model_run]

        return {
            "variant": params["variant"][model_run],
            "health_status_adjustment": params["health_status_adjustment"][model_run],
            "seed": params["seeds"][model_run],
            **{
                k: get_param_value(params[k])
                for k in [
                    "expat",
                    "repat_local",
                    "repat_nonlocal",
                    "non-demographic_adjustment",
                    "inpatient_factors",
                    "outpatient_factors",
                    "aae_factors",
                    "bed_occupancy",
                    "theatres",
                ]
            },
        }

    def _run(
        self,
        rng: np.random.Generator,
        data: pd.DataFrame,
        run_params: dict,
        demo_f: pd.Series,
        hsa_f: pd.Series,
    ) -> tuple[dict, pd.DataFrame]:
        """Run the model

        :param rng: a random number generator created for each model iteration
        :type rng: numpy.random.Generator
        :param data: the DataFrame that we are updating
        :type data: pandas.DataFrame
        :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
        :type run_params: dict
        :param demo_f: the demographic adjustment factors
        :type demo_f: pandas.Series
        :param hsa_f: the health status adjustment factors
        :type hsa_f: pandas.Series

        :returns: a tuple containing the change factors DataFrame and the mode results DataFrame
        :rtype: (dict, pandas.DataFrame)
        """
        # implemeneted by the concrete class

    def run(self, model_run: int) -> tuple[dict, pd.DataFrame]:
        """Run the model once

        Performs a single iteration of the model. The `model_run` parameter controls what parameters
        to use for this run of the model, with the principal model run having 0, and all other model
        runs having a value greater than 0.

        Running the model multiple times with the same `model_run` will give the same results.

        The return value is a tuple that contains the change factors as a dictionary, and the model
        results as a :class:`pandas.DataFrame`.

        :param model_run: the current model run
        :type model_run: int
        :returns: a tuple of the change factors and the model results
        :rtype: (dict, pandas.DataFrame)
        """
        # get the run params
        run_params = self._get_run_params(model_run)
        rng = np.random.default_rng(run_params["seed"])
        data = self.data.copy()
        # demographics factor
        demo_f = self._demog_factors[run_params["variant"]]
        # hsa
        hsa_f = self._health_status_adjustment(data, run_params)
        # choose which function to use
        return self._run(rng, data, run_params, demo_f, hsa_f)

    def aggregate(self, model_results: pd.DataFrame, model_run: int):
        """Aggregate the model results

        Can also be used to aggregate the baseline data by passing in the raw data

        :param model_results: a DataFrame containing the results of a model iteration
        :type model_results: pandas.DataFrame
        :param model_run: the current model run
        :type model_run: int

        :returns: a dictionary containing the different aggregations of this data
        :rtype: dict
        """
        # implemeneted by the concrete class

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
            ["pod"] + (["measure"] if include_measure else []) + (cols if cols else [])
        )
        result = namedtuple("results", cols)
        agg = model_results.groupby(cols)["value"].sum()
        agg.index.names = agg.index.names[:-1] + ["measure"]

        return {name: {result(*k): v for k, v in agg.iteritems()}}
