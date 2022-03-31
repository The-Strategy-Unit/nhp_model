"""
Model Module

Implements the generic class for all other model types.
"""

import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from model.helpers import inrange, rnorm


class Model:  # pylint: disable=too-many-instance-attributes
    """
    Inpatients Model

    * results_path: where the data is stored

    This is a generic implementation of the model. Specific implementations of the model inherit
    from this class.

    In order to run the model you need to pass the path to where the parameters json file is stored.
    This path should be of the form: {dataset_name}/{results}/{model_name}/{model_run_datetime}

    Once the object is constructed you can call either m.run() to run the model and return the data,
    or m.save_run() to run the model and save the results.

    You can also call m.multi_model_run() to run multiple iterations of the model in parallel,
    saving the results to disk to use later.
    """

    def __init__(self, model_type, results_path, columns_to_load=None):
        self._model_type = model_type
        # load the parameters file
        with open(f"{results_path}/params.json", "r", encoding="UTF-8") as params_file:
            self._params = json.load(params_file)
        # store the path where the data is stored and the results are stored
        self._path = str(Path(results_path).parent.parent.parent)
        self._results_path = results_path
        #
        # load the data that's shared across different model types
        #
        with open(f"{self._path}/hsa_gams.pkl", "rb") as hsa_pkl:
            self._hsa_gams = pickle.load(hsa_pkl)
        self._load_demog_factors()
        # load the data. we only need some of the columns for the model, so just load what we need
        data = (
            self._load_parquet(self._model_type, columns_to_load)
            # merge the demographic factors to the data
            .merge(
                self._demog_factors, left_on=["age", "sex"], right_index=True
            ).groupby(["variant"])
        )
        # we now store the data in a dictionary keyed by the population variant
        self._data = {
            k: v.drop(["variant"], axis="columns").set_index(["rn"])
            for k, v in tuple(data)
        }
        # generate the run parameters
        self._generate_run_params()

    #
    def _select_variant(self, rng):
        """
        Randomly select a single variant to use for a model run
        """
        variant = rng.choice(self._variants, p=self._probabilities)
        return (variant, self._data[variant])

    #
    def _load_demog_factors(self):
        dfp = self._params["demographic_factors"]
        start_year = dfp.get("start_year", "2018")
        end_year = dfp.get("end_year", "2043")
        #
        demog_factors = pd.read_csv(os.path.join(self._path, dfp["file"]))
        demog_factors[["age", "sex"]] = demog_factors[["age", "sex"]].astype(int)
        demog_factors["factor"] = demog_factors[end_year] / demog_factors[start_year]
        self._demog_factors = demog_factors.set_index(["age", "sex"])[
            ["variant", "factor"]
        ]
        #
        self._variants = list(dfp["variant_probabilities"].keys())
        self._probabilities = list(dfp["variant_probabilities"].values())

    #
    def _health_status_adjustment(self, data, run_params):
        params = self._params["health_status_adjustment"]
        #
        ages = np.arange(params["min_age"], params["max_age"] + 1)
        adjusted_ages = ages - run_params["health_status_adjustment"]
        hsa = pd.concat(
            [
                pd.DataFrame(
                    {
                        "hsagrp": a,
                        "sex": int(s),
                        "age": ages,
                        "hsa_f": g.predict(adjusted_ages) / g.predict(ages),
                    }
                )
                for (a, s), g in self._hsa_gams.items()
            ]
        )
        return (
            (
                data.reset_index()
                .merge(hsa, on=["hsagrp", "sex", "age"], how="left")
                .set_index(["rn"])
            )["hsa_f"]
            .fillna(1)
            .to_numpy()
        )

    #
    def _load_parquet(self, file, *args):
        """
        Load a parquet file from the file path created by the constructor.

        You can selectively load columns by passing an array of column names to *args.
        """
        return pq.read_pandas(
            os.path.join(self._path, f"{file}.parquet"), *args
        ).to_pandas()

    #
    @staticmethod
    def _factor_helper(data, params, column_values):
        factor = {k: [v] * len(params) for k, v in column_values.items()}
        factor["hsagrp"] = [f"aae_{k}" for k in params.keys()]
        factor["f"] = params.values()
        factor = pd.DataFrame(factor)
        return (
            data.merge(factor, how="left", on=list(factor.columns[:-1]))
            .f.fillna(1)
            .to_numpy()
        )

    #
    def save_run(self, model_run):
        """
        Save the model run and run parameters

        * model_run: the number of this model run

        returns: a tuple containing the time to run the model, and the time to save the results
        """
        start = time()
        mr_data = self.run(model_run)
        end_mr = time()
        principal_change_factors = None
        if isinstance(mr_data, tuple):
            principal_change_factors, mr_data = mr_data
        #
        results_path = f"{self._results_path}/{self._model_type}/model_run={model_run}"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        if principal_change_factors is not None:
            # handle encoding of numpy values (source: https://stackoverflow.com/a/65151218/4636789)
            def np_encoder(obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                return obj

            with open(
                f"{self._results_path}/{self._model_type}_principal_change_factors.json",
                "w",
                encoding="UTF-8",
            ) as pcf:
                json.dump(principal_change_factors, pcf, indent=2, default=np_encoder)
        #
        mr_data.to_parquet(f"{results_path}/{model_run}.parquet")
        #
        return (end_mr - start, time() - end_mr)

    #
    def batch_save(self, model_run, batch_size):
        """
        Run a batch of model runs and save the results
        """
        return [self.save_run(x) for x in range(model_run, model_run + batch_size)]

    #
    def multi_model_runs(self, run_start, model_runs, n_cpus=1, batch_size=4):
        """
        Run multiple model runs in parallel
        """
        print(f"{model_runs} {self._model_type} model runs on {n_cpus} cpus")
        pbar = tqdm(total=model_runs)
        with Pool(n_cpus) as pool:
            results = [
                pool.apply_async(
                    self.batch_save,
                    (i, batch_size if i + batch_size <= model_runs else model_runs - i),
                    callback=lambda _: pbar.update(batch_size),
                )
                for i in range(run_start, run_start + model_runs, batch_size)
            ]
            pool.close()
            pool.join()
            pbar.close()
        #
        times = np.concatenate([r.get() for r in results])
        run_times = [t[0] for t in times]
        save_times = [t[1] for t in times]
        total_times = [t[0] + t[1] for t in times]
        display_times = (
            lambda t: f"mean: {np.mean(t):.3f}, range: [{np.min(t):.3f}, {np.max(t):.3f}]"
        )
        print(
            "",
            "Timings:",
            "=" * 80,
            f"* model runs:    {display_times(run_times)}",
            f"* save results:  {display_times(save_times)}",
            f"* total elapsed: {display_times(total_times)}",
            "=" * 80,
            sep=os.linesep,
        )

    #
    def _generate_run_params(self):
        # the principal projection will have run number 0. all other runs start indexing from 1
        run_params_path = f"{self._results_path}/run_params.json"
        # load the params if they have previously been created
        if os.path.exists(run_params_path):
            with open(run_params_path, "r", encoding="UTF-8") as run_params_file:
                self._run_params = json.load(run_params_file)
                return
        #
        params = self._params
        rng = np.random.default_rng(params["seed"])
        model_runs = params["model_runs"] + 1  # add 1 for the principal

        def gen_value(model_run, i, j):
            if model_run == 0:
                return (
                    i + j
                ) / 2  # for the principal projection, just use the midpoint of the interval
            return rnorm(rng, i, j)  # otherwise,

        #
        self._run_params = {
            # for the principal run, select the most probable variant
            "variant": [self._variants[np.argmax(self._probabilities)]]
            + rng.choice(
                self._variants, model_runs - 1, p=self._probabilities
            ).tolist(),
            "seeds": rng.integers(0, 65535, model_runs).tolist(),
            "health_status_adjustment": [
                [gen_value(m, *i) for m in range(model_runs)]
                for i in params["health_status_adjustment"]["intervals"]
            ],
            "waiting_list_adjustment": params["waiting_list_adjustment"],
            **{
                k0: {
                    k1: {
                        k2: [
                            inrange(
                                gen_value(m, *v2["interval"]), *v2.get("range", [0, 1])
                            )
                            for m in range(model_runs)
                        ]
                        for k2, v2 in v1.items()
                    }
                    for k1, v1 in params[k0].items()
                }
                for k0 in ["strategy_params", "outpatient_factors", "aae_factors"]
            },
        }
        # save the run params
        with open(run_params_path, "w", encoding="UTF-8") as run_params_file:
            json.dump(self._run_params, run_params_file)

    #
    def _get_run_params(self, model_run):
        """
        Gets the parameters for a particular model run
        """
        params = self._run_params
        return {
            "variant": params["variant"][model_run],
            "health_status_adjustment": [
                v[model_run] for v in params["health_status_adjustment"]
            ],
            "seed": params["seeds"][model_run],
            **{
                k0: {
                    k1: {k2: v2[model_run] for k2, v2 in v1.items()}
                    for k1, v1 in params[k0].items()
                }
                for k0 in ["strategy_params", "outpatient_factors", "aae_factors"]
            },
        }

    def _run(self, rng, data, run_params, hsa_f):
        """
        Model Run

        To be implemented by specific classes
        """

    def _principal_projection(self, rng, data, run_params, hsa_f):
        """
        Principal Projection Run

        To be implemented in specific classes
        """

    #
    def run(self, model_run):
        """
        Run the model once

        returns: a tuple of the selected varient and the updated DataFrame
        """
        # get the run params
        run_params = self._get_run_params(model_run)
        rng = np.random.default_rng(run_params["seed"])
        data = self._data[run_params["variant"]]
        # hsa
        hsa_f = self._health_status_adjustment(data, run_params)
        # choose which function to use
        if model_run == 0 and self._model_type == "ip":
            run_fn = self._principal_projection
        else:
            run_fn = self._run
        return run_fn(rng, data, run_params, hsa_f)
