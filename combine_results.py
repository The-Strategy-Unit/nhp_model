"""
Combine Model Results

Each model run creates it's own folder containing the results for that model run. The functions in
this script combine all of the model runs for a dataset/scenario/create_datetime into single files
that are easier to work with in Azure Synapse Analytics
"""

import argparse
import os
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def _combine_results(result_type, data_format, dataset, scenario, create_datetime):
    path = lambda type: os.path.join(
        "results",
        result_type,
        f"activity_type={type}",
        f"dataset={dataset}",
        f"scenario={scenario}",
        f"create_datetime={create_datetime}",
    )

    partitioning = ds.HivePartitioning(pa.schema([("model_run", pa.int16())]))

    def load(activity_type):
        dataset = ds.dataset(
            path(activity_type),
            format=data_format,
            partitioning=partitioning,
        )
        dts = dataset.to_table().to_pandas()
        # Synapse doesn't support hive partitioned columns in external tables
        dts["dataset"] = dataset
        dts["activity_type"] = activity_type
        dts["scenario"] = scenario
        dts["create_datetime"] = create_datetime
        return dts

    activity_types = ["ip", "op", "aae"]
    all_data = pd.concat(load(t) for t in activity_types)

    results_path = path("").replace("activity_type=", "combined")
    os.makedirs(results_path, exist_ok=True)

    if data_format == "parquet":
        all_data.to_parquet(f"{results_path}/0.parquet")
    else:
        all_data.to_csv(f"{results_path}/0.csv", index=False)

    # remove the individual model run folders
    for mrf in [
        os.path.join(path(t), i)
        for t in activity_types
        for i in os.listdir(path(t))
        if i.startswith("model_run")
    ]:
        shutil.rmtree(mrf)


def _combine_model_results(activity_type, dataset, scenario, create_datetime):
    path = os.path.join(
        "results",
        "model_results",
        f"activity_type={activity_type}",
        f"dataset={dataset}",
        f"scenario={scenario}",
        f"create_datetime={create_datetime}",
    )
    model_runs = [i for i in os.listdir(path) if i.startswith("model_run")]

    partitioning = ds.HivePartitioning(pa.schema([("model_run", pa.int16())]))

    dataset = ds.dataset(path, format="parquet", partitioning=partitioning)
    dts = dataset.to_table().to_pandas()

    dts.to_parquet(f"{path}/combined.parquet")

    # remove the individual model run folders
    for mrf in model_runs:
        shutil.rmtree(os.path.join(path, mrf))


def combine(dataset, scenario, create_datetime):
    """Combine model run files into single files

    * dataset: name of the dataset
    * scenario: name of the scenario
    * create_datetime: timestamp of when the model was run

    To be run immediately after the model has finished running, this will take all of the individual
    model run files and combine into a single file.
    """
    _combine_results(
        "aggregated_results", "parquet", dataset, scenario, create_datetime
    )
    _combine_results("change_factors", "csv", dataset, scenario, create_datetime)
    for i in ["op", "aae", "ip"]:
        _combine_model_results(i, dataset, scenario, create_datetime)


def main():
    """
    Main method

    Runs when __name__ == "__main__"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="The name of the dataset")
    parser.add_argument("scenario", help="The name of the scenario")
    parser.add_argument(
        "create_datetime", help="The timestamp of when the model was run"
    )

    # grab the Arguments
    args = parser.parse_args()
    # run combine
    combine(args.dataset, args.scenario, args.create_datetime)


if __name__ == "__main__":
    main()
