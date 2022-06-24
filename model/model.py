"""
Model Module

Implements the generic class for all other model types.
"""

import os
import pickle
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from model.helpers import age_groups, inrange, rnorm


class Model:
    """
    Model

    * model_type: the type of model, either "aae", "ip", or "op"
    * params: the parameters to run the model with
    * data_path: the path to where the data files live
    * columns_to_load [optional]: a list of columns to load from the data

    This is a generic implementation of the model. Specific implementations of the model inherit
    from this class.
    """

    def __init__(
        self,
        model_type: str,
        params: dict,
        data_path: str,
        columns_to_load: list[str] = None,
    ):
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
        self._data_path = os.path.join(data_path, self.params["input_data"])
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

    def _load_demog_factors(self):
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

    def _health_status_adjustment(self, data, run_params):
        lep = self.params["life_expectancy"]
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

    def _load_parquet(self, file, *args):
        """
        Load a parquet file from the file path created by the constructor.

        You can selectively load columns by passing an array of column names to *args.
        """
        return pq.read_pandas(
            os.path.join(self._data_path, f"{file}.parquet"), *args
        ).to_pandas()

    def _factor_helper(self, data, params, column_values):
        hsagrps = (
            data.merge(pd.DataFrame(column_values, index=[0])).hsagrp.unique().tolist()
        )
        factor = {
            h: v
            for k, v in params.items()
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
        params = self.params

        # the principal projection will have run number 0. all other runs start indexing from 1
        rng = np.random.default_rng(params["seed"])
        model_runs = params["model_runs"] + 1  # add 1 for the principal

        def gen_value(model_run, i):
            # we haven't received an interval, just a single value
            if not isinstance(i, list):
                return i
            # for the principal projection, just use the midpoint of the interval
            if model_run == 0:
                return sum(i) / 2
            return rnorm(rng, *i)  # otherwise

        self.run_params = {
            # for the principal run, select the most probable variant
            "variant": [self._variants[np.argmax(self._probabilities)]]
            + rng.choice(
                self._variants, model_runs - 1, p=self._probabilities
            ).tolist(),
            "seeds": rng.integers(0, 65535, model_runs).tolist(),
            "health_status_adjustment": [
                gen_value(m, params["health_status_adjustment"])
                for m in range(model_runs)
            ],
            "waiting_list_adjustment": params["waiting_list_adjustment"],
            "non-demographic_adjustment": {
                k1: {
                    k2: [gen_value(m, v2) for m in range(model_runs)]
                    for k2, v2 in v1.items()
                }
                for k1, v1 in params["non-demographic_adjustment"].items()
            },
            **{
                k0: {
                    k1: {
                        k2: [
                            inrange(
                                gen_value(m, v2["interval"]), *v2.get("range", [0, 1])
                            )
                            for m in range(model_runs)
                        ]
                        for k2, v2 in v1.items()
                    }
                    for k1, v1 in params[k0].items()
                }
                for k0 in ["strategy_params", "outpatient_factors", "aae_factors"]
            },
            "bed_occupancy": {
                k0: {
                    k1: [inrange(gen_value(m, v1)) for m in range(model_runs)]
                    for k1, v1 in params["bed_occupancy"][k0].items()
                }
                for k0 in params["bed_occupancy"].keys()
                if k0 != "specialty_mapping"
            },
        }

    def _get_run_params(self, model_run):
        """
        Gets the parameters for a particular model run
        """
        params = self.run_params
        return {
            "variant": params["variant"][model_run],
            "health_status_adjustment": params["health_status_adjustment"][model_run],
            "seed": params["seeds"][model_run],
            **{
                k0: {
                    k1: {k2: v2[model_run] for k2, v2 in v1.items()}
                    for k1, v1 in params[k0].items()
                }
                for k0 in [
                    "non-demographic_adjustment",
                    "strategy_params",
                    "outpatient_factors",
                    "aae_factors",
                    "bed_occupancy",
                ]
            },
        }

    def _run(
        self,
        rng: np.random.Generator,
        data: pd.DataFrame,
        run_params: dict,
        aav_f: pd.Series,
        hsa_f: pd.Series,
    ):
        """Run the model

        to be implemented by the specific concrete classes
        """

    def run(self, model_run):
        """
        Run the model once

        returns: a tuple of the selected varient and the updated DataFrame
        """
        # get the run params
        run_params = self._get_run_params(model_run)
        rng = np.random.default_rng(run_params["seed"])
        data = self.data.copy()
        # demographics factor
        aav_f = self._demog_factors[run_params["variant"]]
        # hsa
        hsa_f = self._health_status_adjustment(data, run_params)
        # choose which function to use
        return self._run(rng, data, run_params, aav_f, hsa_f)

    def aggregate(self, model_results, model_run):
        """
        Aggregate the model results

        To be implemented by specific classes
        """

    @staticmethod
    def _create_agg(model_results, cols=None, name=None, include_measure=True):
        if name is None:
            name = "+".join(cols) if cols else "default"
        cols = (
            ["pod"] + (["measure"] if include_measure else []) + (cols if cols else [])
        )
        result = namedtuple("results", cols)
        agg = model_results.groupby(cols)["value"].sum()
        agg.index.names = agg.index.names[:-1] + ["measure"]

        return {name: {result(*k): v for k, v in agg.iteritems()}}
