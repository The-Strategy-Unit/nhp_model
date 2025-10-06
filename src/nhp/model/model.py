"""Model Module.

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
import pandas as pd

from nhp.model.data import Data
from nhp.model.health_status_adjustment import (
    HealthStatusAdjustment,
    HealthStatusAdjustmentInterpolated,
)
from nhp.model.helpers import inrange, load_params, rnorm
from nhp.model.model_iteration import ModelIteration, ModelRunResult


class Model:
    """Model.

    This is a generic implementation of the model. Specific implementations of the model inherit
    from this class.

    Args:
        model_type: The type of model, either "aae", "ip", or "op".
        measures: The names of the measures in the model.
        params: The parameters to run the model with, or the path to a params file to load.
        data: A callable that creates a Data instance.
        hsa: An instance of the HealthStatusAdjustment class. If left as None an instance is
            created. Defaults to None.
        run_params: The parameters to use for each model run. Generated automatically if left as
            None. Defaults to None.
        save_full_model_results: Whether to save the full model results or not. Defaults to False.

    Attributes:
        model_type: A string describing the type of model, must be one of "aae", "ip", or "op".
        params: The parameters that are to be used by this model instance.
        data: The HES data extract used by this model instance.

    Example:
        >>> from nhp.model.aae import AaEModel
        >>> from nhp.model.helpers import load_params
        >>> params = load_params("params-sample.json")
        >>> m = AaEModel(params, data)
        >>> model_iteration = ModelIteration(m, 0)
        >>> aggregated_results, step_counts = model_iteration.get_aggregate_results()
    """

    def __init__(
        self,
        model_type: str,
        measures: List[str],
        params: dict,
        data: Callable[[int, str], Data],
        hsa: Any | None = None,
        run_params: dict | None = None,
        save_full_model_results: bool = False,
    ) -> None:
        """Initialise the Model.

        Args:
            model_type: A string saying what type of model this is.
            measures: The names of the measures in this model type.
            params: The parameters to use.
            data: A method to create a Data instance.
            hsa: Health Status Adjustment object. Defaults to None.
            run_params: The run parameters to use. Defaults to None.
            save_full_model_results: Whether to save full model results. Defaults to False.
        """
        valid_model_types = ["aae", "ip", "op"]
        assert model_type in valid_model_types, "Model type must be one of 'aae', 'ip', or 'op'"
        self.model_type = model_type
        if isinstance(params, str):
            params = load_params(params)
        self.params = params
        # add model runtime if it doesn't exist
        if "create_datetime" not in self.params:
            self.params["create_datetime"] = f"{datetime.now():%Y%m%d_%H%M%S}"

        data_loader = data(params["start_year"], params["dataset"])

        self._measures = measures
        # load the data. we only need some of the columns for the model, so just load what we need
        self._load_data(data_loader)
        self._load_strategies(data_loader)
        self._load_demog_factors(data_loader)
        self._load_inequalities_factors(data_loader)
        # create HSA object if it hasn't been passed in
        year = params["start_year"]
        self.hsa = hsa or HealthStatusAdjustmentInterpolated(data_loader, str(year))
        # generate the run parameters if they haven't been passed in
        self.run_params = run_params or self.generate_run_params(params)
        self.save_full_model_results = save_full_model_results

    def _add_pod_to_data(self) -> None:
        """Adds the POD column to data."""
        # to be implemented in ip/op/aae

    def _add_ndggrp_to_data(self) -> None:
        self.data["ndggrp"] = self.data["group"]

    @property
    def measures(self) -> List[str]:
        """The names of the measure columns.

        Returns:
            The names of the measure columns.
        """
        return self._measures

    def _get_data(self, data_loader: Data) -> pd.DataFrame:
        """Load the data."""
        # to be implemented by the concrete classes
        raise NotImplementedError("Subclasses must implement this method")

    def _load_data(self, data_loader: Data) -> None:
        self.data = self._get_data(data_loader).sort_values("rn")
        self.baseline_counts = self.get_data_counts(self.data)

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

    def _load_strategies(self, data_loader: Data) -> None:
        """Load a set of strategies."""
        # to be implemented by the concrete classes
        self.strategies: dict[str, pd.DataFrame] = dict()

    def _load_demog_factors(self, data_loader: Data) -> None:
        """Load the demographic factors.

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
        years = (np.arange(self.params["start_year"], self.params["end_year"]) + 1).astype("str")

        merge_cols = ["age", "sex"]

        def load_factors(factors):
            factors[merge_cols] = factors[merge_cols].astype(int)
            factors = factors.set_index(["variant", *merge_cols])

            return factors[years].apply(lambda x: x / factors[start_year])

        self.demog_factors = load_factors(data_loader.get_demographic_factors())
        self.birth_factors = load_factors(data_loader.get_birth_factors())

    def _load_inequalities_factors(self, data_loader: Data) -> None:
        """Load the inequalities factors.

        Creates 1 private variable:
        * | `self.inequalities_factors`: a pandas.DataFrame containing the inequalities
          | factors to be used for the model iterations, given model parameters
        """
        inequalities_params = pd.DataFrame(
            [
                {"inequality_type": k, "sushrg_trimmed": v}
                for k, hrgs in self.params["inequalities"].items()
                for v in hrgs
            ],
            columns=["inequality_type", "sushrg_trimmed"],
        )

        inequalities_df = (
            data_loader.get_inequalities()
            .drop(columns=["activity_rate", "fitted_line"])
            .melt(
                id_vars=["icb", "sushrg_trimmed", "imd_quintile"],
                var_name="inequality_type",
                value_name="factor",
            )
        )

        self.inequalities_factors = inequalities_df.merge(
            inequalities_params, on=["sushrg_trimmed", "inequality_type"], how="inner"
        ).drop(columns=["inequality_type"])

    @staticmethod
    def generate_run_params(params):
        """Generate the values for each model run from the params.

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
        def gen_value(model_run, i: list[float] | float):
            # we haven't received an interval, just a single value
            if not isinstance(i, list):
                return i
            # for the baseline run, no params need to be set
            if model_run == 0:
                return 1
            lo, hi = i
            return rnorm(rng, lo, hi)  # otherwise

        # function to traverse a dictionary until we find the values to generate from
        def generate_param_values(prm, valid_range):
            if isinstance(prm, dict):
                if "interval" not in prm:
                    return {k: generate_param_values(v, valid_range) for k, v in prm.items()}
                prm = prm["interval"]
            return [valid_range(gen_value(mr, prm)) for mr in range(model_runs)]

        variants = list(params["demographic_factors"]["variant_probabilities"].keys())
        probabilities = list(params["demographic_factors"]["variant_probabilities"].values())
        # there can be numerical inacurracies in the probabilities not summing to 1
        # this will adjust the probabiliteies to sum correctly
        probabilities = [p / sum(probabilities) for p in probabilities]

        variants = [
            variants[np.argmax(probabilities)],
            *rng.choice(variants, model_runs - 1, p=probabilities).tolist(),
        ]

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
            "non-demographic_adjustment": {
                k: generate_param_values(v, inrange_0_5)
                for k, v in params["non-demographic_adjustment"]["values"].items()
            },
            # generate param values for the different items in params: this will traverse the dicts
            # until a value is reached that isn't a dict. Then it will generate the required amount
            # of values for that parameter
            **{
                k: generate_param_values(params[k], v)
                for k, v in [
                    ("waiting_list_adjustment", inrange_0_5),
                    ("expat", inrange_0_1),
                    ("repat_local", inrange_1_5),
                    ("repat_nonlocal", inrange_1_5),
                    ("baseline_adjustment", inrange_0_5),
                    ("activity_avoidance", inrange_0_1),
                    ("efficiencies", inrange_0_1),
                ]
            },
        }

    def _get_run_params(self, model_run: int) -> dict:
        """Get the parameters for a particular model run.

        Takes the run_params and extracts a single item from the arrays for the current model run.

        Args:
            model_run: The current model run.

        Returns:
            The run parameters for the given model run.
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

    def get_data_counts(self, data: pd.DataFrame) -> np.ndarray:
        """Get row counts of data.

        Args:
            data: The data to get the counts of.

        Returns:
            The counts of the data, required for activity avoidance steps.
        """
        raise NotImplementedError()

    def activity_avoidance(
        self, data: pd.DataFrame, model_iteration: ModelIteration
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Perform the activity avoidance (strategies)."""
        # if there are no items in params for activity_avoidance then exit
        if not (params := model_iteration.run_params["activity_avoidance"][self.model_type]):
            return data, None

        rng = model_iteration.rng

        strategies = self.strategies["activity_avoidance"]
        # decide whether to sample a strategy for each row this model run
        strategies = strategies.loc[
            rng.binomial(1, strategies["sample_rate"]).astype("bool"), "strategy"  # type: ignore
        ]
        # join the parameters, then pivot wider
        strategies = (
            strategies.reset_index()
            .merge(pd.Series(params, name="aaf"), left_on="strategy", right_index=True)
            .pivot_table(index="rn", columns="strategy", values="aaf")
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
            model_iteration.fix_step_counts(
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
        """Calculate the rows that have been avoided.

        Args:
            data: The data before the binomial thinning step.
            data_resampled: The data after the binomial thinning step.

        Returns:
            The data that was avoided in the binomial thinning step.
        """
        raise NotImplementedError()

    def go(self, model_run: int) -> ModelRunResult:
        """Run the model and get the aggregated results.

        Needed for running in a multiprocessing pool as you need a serializable method.

        Args:
            model_run: The model run number we want to run.

        Returns:
            A tuple containing a dictionary of results, and the step counts.
        """
        mr = ModelIteration(self, model_run)

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
        """Get aggregation from model results.

        Args:
            results: The results of a model run.
            *args: Any columns you wish to aggregate on.

        Returns:
            Aggregated results.
        """
        return results.groupby(["pod", "sitetret", *args, "measure"])["value"].sum()

    def save_results(self, model_iteration: ModelIteration, path_fn: Callable[[str], str]) -> None:
        """Save the results of running the model.

        This method is used for saving the results of the model run to disk as a parquet file.
        It saves just the `rn` (row number) column and the measure columns, with the intention that
        you rejoin to the original data.

        Args:
            model_iteration: An instance of the ModelIteration class.
            path_fn: A function which takes the activity type and returns a path.
        """
        # implemented by concrete classes

    def apply_resampling(self, row_samples: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
        """Apply row resampling.

        Called from within `model.activity_resampling.ActivityResampling.apply_resampling`.

        Args:
            row_samples: [1xn] array, where n is the number of rows in `data`, containing the new
                values for the measure columns.
            data: The data that we want to update.

        Returns:
            The updated data.
        """
        raise NotImplementedError()

    def aggregate(self, model_iteration: ModelIteration) -> dict[str, pd.Series]:
        """Aggregate the model results.

        Can also be used to aggregate the baseline data by passing in a `ModelIteration` with
        the `model_run` argument set to `-1`.

        Args:
            model_iteration: An instance of the ModelIteration class.

        Returns:
            A dictionary containing the aggregated results.
        """
        model_results = self.process_results(model_iteration.get_model_results())

        base_aggregations = {
            "default": self.get_agg(model_results),
            "sex+age_group": self.get_agg(model_results, "sex", "age_group"),
            "age": self.get_agg(model_results, "age"),
        }

        return {**base_aggregations, **self.specific_aggregations(model_results)}

    def process_results(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data into a format suitable for aggregation in results files.

        Args:
            data: Data to be processed. Format should be similar to Model.data.

        Returns:
            Processed results.
        """
        raise NotImplementedError()

    def specific_aggregations(self, model_results: pd.DataFrame) -> dict[str, pd.Series]:
        """Create other aggregations specific to the model type.

        Args:
            model_results: The results of a model run.

        Returns:
            Dictionary containing the specific aggregations.
        """
        raise NotImplementedError()
