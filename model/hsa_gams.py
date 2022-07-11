"""
Health Status Adjustment GAMs

This file is used to generate the GAMs for Health Status Adjustment.
"""

import os
import pickle
import sys
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq
from janitor import complete  # pylint: disable=unused-import
from pygam import GAM


def _create_activity_type_gams(
    pop: pd.DataFrame,
    path_fn: Callable,
    data: dict,
    gams: dict,
    activity_type: str,
    ignored_hsagrps: list[str] = None,
) -> dict:
    """helper function for generating gams

    * pop: a pandas dataframe containing the population by age/sex
    * path_fn: a fuction that takes a filename and returns a path to that file (for loading datafiles)
    * data: a dictionary that will store the loaded data
    * gams: a dictionary that will store the created gams
    * activity_type: either "ip", "op", or "aae"
    * ignored_hsagrps: an array containing any hsa groups that need to be ignored

    returns: None (updates data and gams dict's)
    """
    dfr = pq.read_pandas(path_fn(f"{activity_type}.parquet")).to_pandas()
    if ignored_hsagrps is not None:
        dfr = dfr[~dfr["hsagrp"].isin(ignored_hsagrps)]
    dfr = dfr[dfr["age"] >= 18]
    dfr = dfr[dfr["age"] <= 90]

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

    data[activity_type] = dfr[["hsagrp", "age", "sex", "activity_rate"]]
    for key, value in tuple(dfr.groupby(["hsagrp", "sex"])):
        gams[key] = GAM().gridsearch(
            value[["age"]].to_numpy(), value["activity_rate"].to_numpy()
        )


def create_gams(dataset: str, base_year: str) -> None:
    """Create GAMs for a dataset

    * dataset: a string to the dataset we want to load
    * base_year: the base year to produce the gams from
    """
    # create a helper function for paths
    def path_fn(filename):
        return os.path.join("data", dataset, filename)

    data = {}
    gams = {}

    # load the population data
    pop = pd.read_csv(path_fn("demographic_factors.csv"))
    pop["age"] = pop["age"].clip(upper=90)
    pop = (
        pop[pop["variant"] == "principal_proj"][["sex", "age", str(base_year)]]
        .rename(columns={str(base_year): "base_year", "age": "age"})
        .groupby(["sex", "age"])
        .agg("sum")
        .reset_index()
    )

    # create the gams
    for activity_type, ignored_hsagrps in [
        ("ip", ["birth", "maternity", "paeds"]),
        ("op", None),
        ("aae", None),
    ]:
        _create_activity_type_gams(
            pop, path_fn, data, gams, activity_type, ignored_hsagrps
        )

    return (pd.concat(data), gams, path_fn)


def main() -> None:
    """Main Method"""
    assert (
        len(sys.argv) == 3
    ), "Must provide exactly 2 argument: the path to the data and the base year"

    _, gams, path_fn = create_gams(sys.argv[1], sys.argv[2])
    # save the gams to disk
    with open(path_fn("hsa_gams.pkl"), "wb") as hsa_pkl:
        pickle.dump(gams, hsa_pkl)


if __name__ == "__main__":
    main()
