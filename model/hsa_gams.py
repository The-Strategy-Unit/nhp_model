"""
Health Status Adjustment GAMs

This file is used to generate the GAMs for Health Status Adjustment.
"""

import os
import pickle
import sys

import pandas as pd
import pyarrow.parquet as pq
from pygam import GAM


def create_gams(path_fn, pop, file, ignored_hsagrps=None):
    """
    Create GAMs
    """
    print(f"Creating gams: {file}")
    dfr = pq.read_pandas(path_fn(f"{file}.parquet")).to_pandas()
    if ignored_hsagrps is not None:
        dfr = dfr[~dfr["hsagrp"].isin(ignored_hsagrps)]
    dfr = dfr[dfr["age"] >= 18]
    dfr = dfr[dfr["age"] <= 90]
    #
    dfr = (
        dfr.groupby(["age", "sex", "hsagrp"])
        .size()
        .reset_index(name="n")
        .complete({"age": range(18, 91)}, "sex", "hsagrp")
        .fillna(0)
        .sort_values(["age", "sex", "hsagrp"])
        .merge(pop, on=["age", "sex"])
    )
    dfr["activity_rate"] = dfr["n"] / dfr["base_year"]
    #
    return {
        k: GAM().gridsearch(v[["age"]].to_numpy(), v["activity_rate"].to_numpy())
        for k, v in tuple(dfr.groupby(["hsagrp", "sex"]))
    }


def main():
    """Main Method"""
    assert (
        len(sys.argv) == 3
    ), "Must provide exactly 2 argument: the path to the data and the base year"
    path, base_year = sys.argv[1:]  # pylint: disable=unbalanced-tuple-unpacking
    # create a helper function for creating paths
    path_fn = lambda f: os.path.join(path, f)
    # load the population data
    pop = pd.read_csv(path_fn("demographic_factors.csv"))
    pop["age"] = pop["age"].clip(upper=90)
    pop = (
        pop[pop["variant"] == "principal"][["sex", "age", str(base_year)]]
        .rename(columns={str(base_year): "base_year", "age": "age"})
        .groupby(["sex", "age"])
        .agg("sum")
        .reset_index()
    )
    # create the gams
    gams = {
        **create_gams(path_fn, pop, "ip", ["birth", "maternity", "paeds"]),
        **create_gams(path_fn, pop, "op"),
        **create_gams(path_fn, pop, "aae"),
    }
    # save the gams to disk
    with open(path_fn("hsa_gams.pkl"), "wb") as hsa_pkl:
        pickle.dump(gams, hsa_pkl)


if __name__ == "__main__":
    main()
