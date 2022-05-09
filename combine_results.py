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
    run_results_path,
    results_path,
    result_type,
    data_format,
    dataset,
    scenario,
    create_datetime,
):  # pylint: disable=too-many-arguments
    print(f"combine_results: {result_type=}")
    path = lambda type: os.path.join(
        run_results_path,
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

    combined_results_path = os.path.join(
        results_path, result_type, f"dataset={dataset}", f"scenario={scenario}"
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
    run_results_path, results_path, activity_type, dataset, scenario, create_datetime
):
    print(f"combine model results: {activity_type=}")
    raw_results_path = os.path.join(
        run_results_path,
        "model_results",
        f"activity_type={activity_type}",
        f"dataset={dataset}",
        f"scenario={scenario}",
        f"create_datetime={create_datetime}",
    )
    if not os.path.exists(raw_results_path):
        return

    combined_results_path = os.path.join(
        results_path, "model_results", f"dataset={dataset}", f"scenario={scenario}"
    )

    partitioning = ds.HivePartitioning(pa.schema([("model_run", pa.int16())]))

    os.makedirs(combined_results_path, exist_ok=True)

    print("* load data")
    (
        ds.dataset(raw_results_path, format="parquet", partitioning=partitioning)
        .to_table()
        .to_pandas()
        .to_parquet(f"{combined_results_path}/combined.parquet")
    )
    print("* done \n")


def _copy_params(
    run_results_path, results_path, params_type, dataset, scenario, create_datetime
):
    path = os.path.join(params_type, f"dataset={dataset}", f"scenario={scenario}")
    filename = f"{create_datetime}.json"
    raw_path = os.path.join(run_results_path, path)
    combined_path = os.path.join(results_path, path)
    os.makedirs(combined_path, exist_ok=True)
    shutil.copy(os.path.join(raw_path, filename), os.path.join(combined_path, filename))


def combine(run_results_path, results_path, dataset, scenario, create_datetime):
    """Combine model run files into single files

    * run_results_path: the path to where the model runs are saved
    * resutls_path: the path where we will save the combined results
    * dataset: name of the dataset
    * scenario: name of the scenario
    * create_datetime: timestamp of when the model was run

    To be run immediately after the model has finished running, this will take all of the individual
    model run files and combine into a single file.

    It will clean up the run results after finishing.
    """
    _combine_results(
        run_results_path,
        results_path,
        "aggregated_results",
        "parquet",
        dataset,
        scenario,
        create_datetime,
    )
    _combine_results(
        run_results_path,
        results_path,
        "change_factors",
        "csv",
        dataset,
        scenario,
        create_datetime,
    )
    for i in ["op", "aae", "ip"]:
        _combine_model_results(
            run_results_path, results_path, i, dataset, scenario, create_datetime
        )
    for i in ["params", "run_params"]:
        _copy_params(
            run_results_path, results_path, i, dataset, scenario, create_datetime
        )
    # we can now clean up the run results
    shutil.rmtree(run_results_path)


def main():
    """
    Main method

    Runs when __name__ == "__main__"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("run_results_path", help="The path to the run results")
    parser.add_argument(
        "results_path", help="The path where to save the combined results"
    )
    parser.add_argument("dataset", help="The name of the dataset")
    parser.add_argument("scenario", help="The name of the scenario")
    parser.add_argument(
        "create_datetime", help="The timestamp of when the model was run"
    )

    # grab the Arguments
    args = parser.parse_args()
    # run combine
    combine(
        args.run_results_path,
        args.results_path,
        args.dataset,
        args.scenario,
        args.create_datetime,
    )


if __name__ == "__main__":
    main()
