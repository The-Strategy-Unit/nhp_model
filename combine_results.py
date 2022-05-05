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


def _combine_results(
    results_path, result_type, data_format, dataset, scenario, create_datetime
):  # pylint: disable=too-many-arguments
    print(f"combine_results: {result_type=}")
    path = lambda type: os.path.join(
        results_path,
        result_type,
        f"activity_type={type}",
        f"dataset={dataset}",
        f"scenario={scenario}",
        f"create_datetime={create_datetime}",
    )

    partitioning = ds.HivePartitioning(pa.schema([("model_run", pa.int16())]))

    def load(activity_type):
        print(f"* load({activity_type})")
        dts = (
            ds.dataset(
                path(activity_type),
                format=data_format,
                partitioning=partitioning,
            )
            .to_table()
            .to_pandas()
        )
        # Synapse doesn't support hive partitioned columns in external tables
        dts["dataset"] = dataset
        dts["activity_type"] = activity_type
        dts["scenario"] = scenario
        dts["create_datetime"] = create_datetime
        return dts

    activity_types = ["ip", "op", "aae"]
    all_data = [load(t) for t in activity_types if os.path.exists(path(t))]
    if len(all_data) == 0:
        print("no data: exiting")
        return

    all_data = pd.concat(all_data)
    all_data["value"] = all_data["value"].astype(int)

    combined_results_path = (
        path("")
        .replace("activity_type=", "combined")
        .replace(f"create_datetime={create_datetime}", "")
    )
    os.makedirs(combined_results_path, exist_ok=True)

    print("* saving data")
    if data_format == "parquet":
        all_data.to_parquet(
            f"{combined_results_path}/{create_datetime}.parquet", index=False
        )
    else:
        all_data.to_csv(f"{combined_results_path}/{create_datetime}.csv", index=False)

    print("* done\n")


def _combine_model_results(
    results_path, activity_type, dataset, scenario, create_datetime
):
    print(f"combine model results: {activity_type=}")
    path = os.path.join(
        results_path,
        "model_results",
        f"activity_type={activity_type}",
        f"dataset={dataset}",
        f"scenario={scenario}",
        f"create_datetime={create_datetime}",
    )
    if not os.path.exists(path):
        return

    partitioning = ds.HivePartitioning(pa.schema([("model_run", pa.int16())]))

    print("* load data")
    (
        ds.dataset(path, format="parquet", partitioning=partitioning)
        .to_table()
        .to_pandas()
        .to_parquet(f"{path}/combined.parquet")
    )
    print("* done \n")


def combine(results_path, dataset, scenario, create_datetime):
    """Combine model run files into single files

    * dataset: name of the dataset
    * scenario: name of the scenario
    * create_datetime: timestamp of when the model was run

    To be run immediately after the model has finished running, this will take all of the individual
    model run files and combine into a single file.
    """
    _combine_results(
        results_path,
        "aggregated_results",
        "parquet",
        dataset,
        scenario,
        create_datetime,
    )
    _combine_results(
        results_path, "change_factors", "csv", dataset, scenario, create_datetime
    )
    for i in ["op", "aae", "ip"]:
        _combine_model_results(results_path, i, dataset, scenario, create_datetime)


def main():
    """
    Main method

    Runs when __name__ == "__main__"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", help="The path to the results")
    parser.add_argument("dataset", help="The name of the dataset")
    parser.add_argument("scenario", help="The name of the scenario")
    parser.add_argument(
        "create_datetime", help="The timestamp of when the model was run"
    )

    # grab the Arguments
    args = parser.parse_args()
    # run combine
    combine(args.results_path, args.dataset, args.scenario, args.create_datetime)


if __name__ == "__main__":
    main()
