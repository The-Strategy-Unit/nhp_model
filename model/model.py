"""
Model Module

Implements the generic class for all other model types.
"""

import json
import os
import pickle
from multiprocessing import Pool

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

    def __init__(
        self, model_type, params_file, data_path, results_path, columns_to_load=None
    ):  # pylint: disable=too-many-arguments
        self._model_type = model_type
        with open(params_file, "r", encoding="UTF-8") as prf:
            self._params = json.load(prf)
        # store the path where the data is stored and the results are stored
        self._data_path = os.path.join(data_path, self._params["input_data"])
        self._results_path = results_path
        #
        # load the data that's shared across different model types
        #
        with open(f"{self._data_path}/hsa_gams.pkl", "rb") as hsa_pkl:
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
        # save the baseline aggregated rows
        self.aggregate_baseline()

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
        demog_factors = pd.read_csv(os.path.join(self._data_path, dfp["file"]))
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
        lep = self._params["life_expectancy"]
        ages = np.tile(np.arange(lep["min_age"], lep["max_age"] + 1), 2)
        sexs = np.repeat([1, 2], len(ages) // 2)
        lex = pd.DataFrame({"age": ages, "sex": sexs, "ex": lep["m"] + lep["f"]})
        lex["ex"] *= run_params["health_status_adjustment"]
        lex["adjusted_age"] = lex["age"] - lex["ex"]

        lex.set_index("sex", inplace=True)

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

    #
    def _load_parquet(self, file, *args):
        """
        Load a parquet file from the file path created by the constructor.

        You can selectively load columns by passing an array of column names to *args.
        """
        return pq.read_pandas(
            os.path.join(self._data_path, f"{file}.parquet"), *args
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

    def _save_path(self, model_run, save_type):
        path = os.path.join(
            self._results_path,
            save_type,
            f"activity_type={self._model_type}",
            f"dataset={self._params['input_data']}",
            f"scenario={self._params['name']}",
            f"create_datetime={self._params['create_datetime']}",
            f"model_run={model_run}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    def aggregate_baseline(self):
        """
        Save the aggregated results for the baseline
        """
        if not self._params.get("aggregate_results", True):
            return
        data = self._data["principal"]
        aggregated_path = self._save_path(-1, "aggregated_results")
        self.aggregate(data).to_parquet(f"{aggregated_path}/0.parquet")

    #
    def save_run(self, model_run):
        """
        Save the model run and run parameters

        * model_run: the number of this model run

        returns: the results of `self.run(model_run)`
        """
        # run the model
        change_factors, mr_data = self.run(model_run)
        # save row level results (if param is not set, defaults to not saving row level results)
        if self._params.get("save_all_results", False):
            results_path = self._save_path(model_run, "model_results")
            mr_data.to_parquet(f"{results_path}/0.parquet")
        # aggregate the results (if param is not set, defaults to saving aggregated results)
        if self._params.get("aggregate_results", True):
            aggregated_path = self._save_path(model_run, "aggregated_results")
            self.aggregate(mr_data).to_parquet(f"{aggregated_path}/0.parquet")
        # Save the change factors, so long as it's not an empty dictionary
        if change_factors is not None:
            change_factors_path = self._save_path(model_run, "change_factors")
            change_factors.to_csv(f"{change_factors_path}/0.csv", index=False)
        return (change_factors, mr_data)

    #
    def batch_save(self, i, j, batch_size):
        """
        Run a batch of model runs and save the results
        """
        return [self.save_run(mr) for mr in range(i, min(j, i + batch_size))]

    #
    def multi_model_runs(self, run_start, model_runs, n_cpus=1, batch_size=4):
        """
        Run multiple model runs in parallel
        """
        pbar = tqdm(total=model_runs)
        with Pool(n_cpus) as pool:
            run_end = run_start + model_runs
            results = [
                pool.apply_async(
                    self.batch_save,
                    (
                        i,
                        run_end,
                        batch_size,
                    ),
                    callback=lambda _: pbar.update(batch_size),
                )
                for i in range(run_start, run_end, batch_size)
            ]
            pool.close()
            pool.join()
            pbar.close()
        # make sure to get the results - if we don't then no errors that occurred will be raised
        print(
            f"Model runs completed: {sum(len(r.get()) for r in results)} / {model_runs}"
        )
        assert sum(len(r.get()) for r in results) == model_runs

    #
    def _generate_run_params(self):
        params = self._params
        path_fn = lambda save_type: os.path.join(
            self._results_path,
            save_type,
            f"dataset={self._params['input_data']}",
            f"scenario={self._params['name']}",
        )
        json_name = f"create_datetime={self._params['create_datetime']}.json"
        # if the params file doesn't exist in it's results folder, save it there now
        params_path = os.path.join(path_fn("params"), json_name)
        if not os.path.exists(params_path):
            os.makedirs(path_fn("params"), exist_ok=True)
            with open(params_path, "w", encoding="UTF-8") as params_file:
                json.dump(params, params_file)
        # load the run params if they have previously been created
        run_params_path = os.path.join(path_fn("run_params"), json_name)
        if os.path.exists(run_params_path):
            with open(run_params_path, "r", encoding="UTF-8") as run_params_file:
                self._run_params = json.load(run_params_file)
                return
        #
        # the principal projection will have run number 0. all other runs start indexing from 1
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
                gen_value(m, *params["health_status_adjustment"])
                for m in range(model_runs)
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
        os.makedirs(path_fn("run_params"), exist_ok=True)
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
            "health_status_adjustment": params["health_status_adjustment"][model_run],
            "seed": params["seeds"][model_run],
            **{
                k0: {
                    k1: {k2: v2[model_run] for k2, v2 in v1.items()}
                    for k1, v1 in params[k0].items()
                }
                for k0 in ["strategy_params", "outpatient_factors", "aae_factors"]
            },
        }

    def _run(
        self, rng, data, run_params, aav_f, hsa_f
    ):  # pylint: disable=too-many-arguments
        """
        Model Run

        To be implemented by specific classes
        """

    def run(self, model_run):
        """
        Run the model once

        returns: a tuple of the selected varient and the updated DataFrame
        """
        # get the run params
        run_params = self._get_run_params(model_run)
        rng = np.random.default_rng(run_params["seed"])
        data = self._data[run_params["variant"]].copy()
        # admission avoidance
        aav_f = data["factor"]
        data.drop(["factor"], axis="columns", inplace=True)
        # hsa
        hsa_f = self._health_status_adjustment(data, run_params)
        # choose which function to use
        return self._run(rng, data, run_params, aav_f, hsa_f)

    def aggregate(self, model_results):
        """
        Aggregate the model results

        To be implemented by specific classes
        """
        return model_results
