"""Model Save Classes

Allows the user to change the way that model results are saved. Currently implements two classes:

* `LocalSave` saves the results to local disk
* `CosmosDBSave` saves the results to Cosmos DB
"""

import json
import os

from azure.cosmos import CosmosClient
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()


class ModelSave:
    """Run the model and save the results

    This is a generic implementation and should not be used  directly"""

    def __init__(self, model):
        self._dataset = model.params["input_data"]
        self._scenario = model.params["name"]
        self._create_datetime = model.params["create_datetime"]
        #
        self._activity_type = model._model_type
        self._model = model


class LocalSave(ModelSave):
    """Save the results locally to disk

    This will save the results and params in a data lake structure to locally available storage
    """

    def __init__(self, model, results_path):
        super().__init__(model)
        self._base_results_path = results_path
        self._results_path = os.path.join(
            f"dataset={self._dataset}",
            f"scenario={self._scenario}",
            f"create_datetime={self._create_datetime}",
        )
        #
        self.save_params()

    def run_model(self, model_run):
        """Run the model and save the results

        * model_run: the iteration of the model we want to run

        Saves the results and change factors of a model run to the path:
            `{results|change_factors}/dataset=.../scenario=.../create_datetime=.../`
        as 0.parquet (for results) and 0.csv (for change_factors).

        Uses a hive partition scheme so it is easy to load the data with pyarrow
        """
        # do nothing for the baseline model run
        if model_run == -1:
            return
        # run the model
        change_factors, results = self._model.run(model_run)

        # function for generating the paths
        path = lambda save_type: os.path.join(
            self._base_results_path,
            save_type,
            f"activity_type={self._activity_type}",
            self._results_path,
            f"{model_run=}",
        )
        # save results
        os.makedirs(mr_path := path("model_results"), exist_ok=True)
        results.to_parquet(f"{mr_path}/0.parquet")
        # save change factors
        os.makedirs(cf_path := path("change_factors"), exist_ok=True)
        change_factors.to_csv(f"{cf_path}/0.csv")

        return None

    def post_runs(self, items):
        """Post running of all model runs

        For local save, this does nothing
        """

    def save_params(self):
        """Save the params items

        Saves the params and run params dictionaries to the path:
            `params/dataset=.../scenario=.../create_datetime=.../`
        as params.json and run_params.json
        """
        path = os.path.join(self._base_results_path, "params", self._results_path)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "params.json"), "w", encoding="UTF-8") as prf:
            json.dump(self._model.params, prf)
        with open(os.path.join(path, "run_params.json"), "w", encoding="UTF-8") as prf:
            json.dump(self._model.run_params, prf)


class CosmosDBSave:
    """Save the results to Azure Cosmos DB

    This will save the results and params into an existing Azure Cosmos DB account
    """

    # note: split the running of models and saving of results in cosmos
    # the database connection isn't multiprocessor safe, so we first run the model runs in parallel
    # then upload the results to cosmos after

    def __init__(self, model, database):
        #
        self._database = database
        #
        self._model = model
        #
        activity_type = model._model_type
        dataset = model.params["input_data"]
        scenario = model.params["name"]
        create_datetime = model.params["create_datetime"]
        #
        self._base_item = {
            "id": f"{dataset}|{scenario}|{create_datetime}|{activity_type}",
            "dataset": dataset,
            "scenario": scenario,
            "create_datetime": create_datetime,
        }
        self._variants = model.run_params["variant"]
        #
        self.save_params()
        baseline = self.run_model(-1)
        self.post_runs([baseline])

    def _get_database_container(self, container):
        client = CosmosClient(os.getenv("COSMOS_ENDPOINT"), os.getenv("COSMOS_KEY"))
        return client.get_database_client(self._database).get_container_client(
            container
        )

    def run_model(self, model_run):
        """Run the model

        * model_run: the iteration of the model we want to run

        Returns the item to be uploaded to Cosmos later

        """
        item = {"model_run": model_run, **self._base_item}
        if model_run == -1:
            variant = self._model.run_params["variant"][0]
            item["results"] = self._model.aggregate(self._model.data[variant])
        else:
            change_factors, results = self._model.run(model_run)
            item["selected_variant"] = self._variants[model_run]
            item["results"] = self._model.aggregate(results)
            item["change_factors"] = change_factors.to_dict(orient="records")

        return (model_run, item)

    def post_runs(self, items):
        """Upload results to Cosmos

        Saves the results and change factors of a model run into the "results" container in the
        Cosmos database.

        Data is stored partitioned by `model_run` and saved with an id built up like:
            `dataset|scenario|create_datetime|activity_type`
        """
        container = self._get_database_container("results")
        for model_run, item in tqdm(items, "Uploading to cosmos", total=len(items)):
            container.upsert_item(item, partition_key=model_run)

    def save_params(self):
        """Save the params items

        Saves the params and run params dictionaries into separate containers in the Cosmos
        database.
        """
        # create connections to the database containers
        params_container = self._get_database_container("params")
        run_params_container = self._get_database_container("run_params")
        # extract the params objects from the model
        params = self._model.params
        run_params = self._model.run_params
        # create an ID to use in the Cosmos item
        run_id = f"{params['input_data']}|{params['name']}|{params['create_datetime']}"
        run_params["id"] = params["id"] = run_id
        # upload the param files to their respective database containers
        params_container.upsert_item(params)
        run_params_container.upsert_item(run_params)
