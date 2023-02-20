"""Model Save Classes

Allows the user to change the way that model results are saved. Currently implements two classes:

* `LocalSave` saves the results to local disk
* `CosmosDBSave` saves the results to Cosmos DB
"""

import json
import os
import shutil
from collections import defaultdict
from tempfile import mkdtemp

import dill
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from azure.cosmos import ContainerProxy, CosmosClient
from dotenv import load_dotenv

from model.model import Model


class ModelSave:
    """Run the model and save the results

    This is a generic implementation and should not be used  directly, instead use one of either
    `CosmosDBSave` or `LocalSave`.

    :param params: the parameters we will use to run the model
    :type params: dict
    :param results_path: the path where we will save the results to
    :type results_path: str
    :param temp_path: the path where we will store store results temporarily. If left as `None`,
        we will choose a temporary path using `os.mkdtemp()`
    :type temp_path: str, optional
    :param save_results: whether to save the full model results, or not (defaults to `False`)
    :type save_results: bool, optional
    """

    def __init__(
        self,
        params: dict,
        results_path: str,
        temp_path: str = None,
        save_results: bool = False,
    ) -> None:
        self._dataset = params["dataset"]
        self._scenario = params["scenario"]
        self._create_datetime = params["create_datetime"]
        #
        self._run_id = f"{self._dataset}__{self._scenario}__{self._create_datetime}"
        #
        self._model_runs = params["model_runs"]
        self._params = params
        self._model = None
        #
        self._base_results_path = results_path
        self._results_path = os.path.join(
            f"dataset={self._dataset}",
            f"scenario={self._scenario}",
            f"create_datetime={self._create_datetime}",
        )
        self._temp_path = temp_path or mkdtemp()
        self._ar_path = os.path.join(self._temp_path, "aggregated_results")
        os.makedirs(self._ar_path, exist_ok=True)
        self._cf_path = os.path.join(self._temp_path, "change_factors")
        os.makedirs(self._cf_path, exist_ok=True)
        #
        self._item_base = {
            "id": self._run_id,
            "dataset": self._dataset,
            "scenario": self._scenario,
            "create_datetime": self._create_datetime,
            "model_runs": self._model_runs,
            "submitted_by": params.get("submitted_by", None),
            "start_year": params["start_year"],
            "end_year": params["end_year"],
            "app_version": params.get("app_version", "0.1"),
        }
        #
        self._save_results = save_results

    def set_model(self, model: Model) -> None:
        """Set the current model

        Updates the current model which we will run using `run_model()`.

        :param model: an instance of `Model`
        :type model: Model
        """
        self._model = model

    def run_model(self, model_run: int) -> None:
        """Run the model and save the results

        Saves the results and change factors of a model run to the path:
            `{results|change_factors}/dataset=.../scenario=.../create_datetime=.../`
        as 0.parquet (for results) and 0.csv (for change_factors).

        Uses a hive partition scheme so it is easy to load the data with pyarrow

        :param model_run: the iteration of the model we want to run
        :type model_run: int
        """
        model = self._model
        activity_type = model.model_type
        # don't run the model if it's the baseline: just run the aggregate step
        if model_run == -1:
            results = model.data.copy()
        else:
            # run the model
            mr = model.run(model_run)
            # save results
            if self._save_results:

                def path_fn(activity_type):
                    p = os.path.join(
                        self._base_results_path,
                        "model_results",
                        f"{activity_type=}",
                        self._results_path,
                        f"{model_run=}",
                    )
                    os.makedirs(p, exist_ok=True)
                    return p

                model.save_results(mr, path_fn)
            # save change factors
            mr.get_step_counts().assign(
                activity_type=activity_type, model_run=model_run
            ).to_parquet(
                f"{self._cf_path}/{activity_type}_{model_run}.parquet", index=False
            )

        # save aggregated results
        aggregated_results = model.aggregate(results, model_run)
        with open(f"{self._ar_path}/{activity_type}_{model_run}.dill", "wb") as arf:
            dill.dump(aggregated_results, arf)

    def post_runs(self) -> None:
        """Post running of all model runs

        Anything that has to happen after all of the model runs have completed running occures here.

        We save out the parameters file that was used, along with the run parameters, before
        removing the temporary path contents.
        """
        # save params
        os.makedirs(pr_path := f"{self._base_results_path}/params", exist_ok=True)
        with open(f"{pr_path}/{self._run_id}.json", "w", encoding="UTF-8") as prf:
            json.dump(self._params, prf)
        # save run params
        os.makedirs(pr_path := f"{self._base_results_path}/run_params", exist_ok=True)
        with open(f"{pr_path}/{self._run_id}.json", "w", encoding="UTF-8") as prf:
            json.dump(self._model.run_params, prf)
        # clean up temporary files
        shutil.rmtree(self._temp_path)

    def _combine_aggregated_results(self) -> dict:
        """Combine aggregate results

        Takes the individual files of the aggregated model results and combines into a single
        dictionary of results

        :returns: a single dictionary of the aggregated model results
        :rtype: dict
        """
        aggregated_results = {}
        for activity_type in ["aae", "ip", "op"]:
            aggregated_results[activity_type] = list()
            for model_run in range(-1, self._model_runs + 1):
                file = f"{self._ar_path}/{activity_type}_{model_run}.dill"
                if not os.path.exists(file):
                    continue
                with open(file, "rb") as arf:
                    aggregated_results[activity_type].append(dill.load(arf))

        flipped_results = self._flip_results({k: v for k, v in aggregated_results.items() if v != []})

        available_aggregations = defaultdict(set)
        for key, val in flipped_results.items():
            for i in val:
                activity_type = i["pod"][:3].replace("_", "")
                available_aggregations[activity_type].add(key)

        return {
            **self._item_base,
            "available_aggregations": {
                k: sorted(list(v)) for k, v in available_aggregations.items()
            },
            "selected_variants": self._model.run_params["variant"],
            "results": flipped_results,
        }

    @staticmethod
    def _flip_results(results: dict) -> dict:
        """Take array of model run results and flip

        Takes an array of aggregate type/aggregate: value, and flips so we have a dictionary
        of aggregate type/aggregates: [value].

        We split out the baseline and principal values, as well as calculating summary statistics
        on the model run values

        :param results: a dictionary containining entries for the activity types and arrays of the
            aggregated results

        :returns: a dictionary that has been inverted
        :rtype: dict
        """
        # create a nested defaultdict which contains an array of 0's the lenght of the results:
        # that gives us a value even if a model run did not output a row for a given aggregate
        flipped = defaultdict(lambda: defaultdict(lambda: [0] * len(results[list(results.keys())[0]])))
        for dataset_results in results.values():
            # iterate over each result
            for idx, result in enumerate(dataset_results):
                # iterate over the aggregate types in each result
                for agg_type, agg_type_v in result.items():
                    # iterate the aggregates inside the aggregate type, use += as some measures
                    # appear in both ip/op and need to be combined
                    for aggregate, aggregate_v in agg_type_v.items():
                        flipped[agg_type][aggregate][idx] += aggregate_v
        # convert our defaultdicts to regular dict's, splitting out the baseline and principal
        # model run results from the monte carlo sim.
        # only keep the full model results if the aggregate type is "default"
        return dict(
            {
                k1: [
                    {
                        **k2._asdict(),  # expand out the namedtuple to a dictionary
                        "baseline": int(v2[0]),
                        "principal": int(v2[1]),
                        "median": np.median(v2[2:]),
                        "lwr_ci": np.quantile(v2[2:], 0.05),
                        "upr_ci": np.quantile(v2[2:], 0.95),
                        **(
                            {"model_runs": [int(vv) for vv in v2[2:]]}
                            if k1 in ["default", "bed_occupancy", "theatres_available"]
                            else {}
                        ),
                    }
                    for k2, v2 in v1.items()
                    if sum(v2) > 0  # skip if this aggregation has no activity
                ]
                for k1, v1 in flipped.items()
            }
        )

    def _combine_change_factors(self) -> pd.DataFrame:
        """Combine Change Factors

        :returns: a single DataFrame of all of the Change Factors
        :rtype: pandas.DataFrame
        """
        return (
            pq.ParquetDataset(self._cf_path, use_legacy_dataset=False)
            .read_pandas()
            .to_pandas()
        )


class LocalSave(ModelSave):
    """Save the model results locally

    Utilises ModelSave to store results to local storage.
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def post_runs(self) -> None:
        """Post running of all model runs

        Performs the same actions as `ModelSave.post_runs`, but also saves the aggregated results
        and the change factors to the `results_path` the object was initialised with.
        """

        # save aggregated results
        os.makedirs(
            ar_path := f"{self._base_results_path}/aggregated_results", exist_ok=True
        )
        with open(f"{ar_path}/{self._run_id}.json", "w", encoding="UTF-8") as arf:
            json.dump(self._combine_aggregated_results(), arf)

        # save change factors
        os.makedirs(
            cf_path := f"{self._base_results_path}/change_factors", exist_ok=True
        )
        self._combine_change_factors().to_csv(
            f"{cf_path}/{self._run_id}.csv", index=False
        )

        # call base method
        super().post_runs()


class CosmosDBSave(ModelSave):
    """Save the model results to Cosmos DB

    Utilises ModelSave to store results to both the file paths and to Cosmos DB.

    Environment variables are used to configure the various secrets to connect to CosmosDB:

    * `COSMOS_ENDPOINT`: should be something like https://my_cosmos_account.documents.azure.com:443/
    * `COSMOS_KEY`: the key used to connect to your CosmosDB account
    * `COSMOS_DB`: the name of the database that you are storing the results in
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)
        #
        load_dotenv()
        self._database = os.getenv("COSMOS_DB")
        self._cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
        self._cosmos_key = os.getenv("COSMOS_KEY")

    def post_runs(self) -> None:
        """Post running of all model runs

        Performs the same actions as `ModelSave.post_runs`, but uploads the results and change
        factors to CosmosDB.
        """
        self._upload_results()
        self._upload_change_factors()
        super().post_runs()

    def _upload_results(self) -> None:
        """Upload the results to CosmosDB

        Takes the results and stores them in the "results" container
        """
        aggregated_results = self._combine_aggregated_results()

        item = {"id": self._run_id, **aggregated_results}

        self._get_database_container("results").upsert_item(
            item, partition_key=self._run_id
        )

    @staticmethod
    def _change_factor_to_dict(change_factor: pd.DataFrame) -> dict:
        """Converts the change factors DataFrame to a dictionary

        :param change_factor: the change factors DataFrame
        :type change_factor: pandas.DataFrame

        :returns: the change factors as a dictionary ready to upload to CosmosDB
        :type: dict
        """
        acf = (
            change_factor.drop(["activity_type", "measure"], axis="columns")
            .set_index(["change_factor", "strategy", "model_run"])
            .unstack(fill_value=0)
            .stack()
            .reset_index()
            .groupby(["change_factor", "strategy"], as_index=False)
            .agg({"value": list})
            .to_dict("records")
        )
        for i in acf:
            if i["strategy"] == "-":
                i.pop("strategy")
            if i["change_factor"] == "baseline":
                i["baseline"] = i["value"].pop(0)
                i.pop("value")
            else:
                i["principal"] = i["value"].pop(0)
        return acf

    def _upload_change_factors(self) -> None:
        """Upload the change factors to CosmosDB

        Takes the changes factors and stores them in the "change_factors" container
        """
        change_factors = self._combine_change_factors()

        item = {
            **self._item_base,
            **{
                d: [
                    {"measure": m, "change_factors": self._change_factor_to_dict(v2)}
                    for m, v2 in tuple(v1.groupby(["measure"]))
                ]
                for d, v1 in tuple(change_factors.groupby(["activity_type"]))
            },
        }

        self._get_database_container("change_factors").upsert_item(
            item, partition_key=self._run_id
        )

    def _get_database_container(self, container: str) -> ContainerProxy:
        """Get a database container

        Helper method to get a CosmosDB container

        :param container: The name of the container
        :type container: str

        :returns: the CosmosDB contianer
        :rtype: azure.cosmos.ContainerProxy
        """
        client = CosmosClient(self._cosmos_endpoint, self._cosmos_key)
        database = client.get_database_client(self._database)
        return database.get_container_client(container)
