import pickle
import pyarrow.parquet as pq
import pandas as pd
import json
import os

from pathlib import Path
from pathos.multiprocessing import ProcessPool

class Model:
  def __init__(self, results_path):
    # load the parameters file
    with open(f"{results_path}/params.json", "r") as f: params = json.load(f)
    self._params = params
    # load the required data
    self._path = str(Path(results_path).parent.parent.parent)
    self._results_path = results_path
    #
    # load the data that's shared across different model types
    #
    with open(f"{self._path}/hsa_gams.pkl", "rb") as f: self._hsa_gams = pickle.load(f)
    # prepare demographic factors
    dfp = params["demographic_factors"]
    start_year = dfp.get("start_year", "2018")
    end_year = dfp.get("end_year", "2043")
    #
    demog_factors = pd.read_csv(os.path.join(self._path, dfp["file"]))
    #
    demog_factors["age"] = demog_factors["age"].astype(int)
    demog_factors["sex"] = demog_factors["sex"].astype(int)
    demog_factors["factor"] = demog_factors[end_year] / demog_factors[start_year]
    self._demog_factors = demog_factors.rename(columns = {"age": "admiage"}).set_index(["admiage", "sex"])[["variant", "factor"]]
    #
    self._variants = list(dfp["variant_probabilities"].keys())
    self._probabilities = list(dfp["variant_probabilities"].values())
  #
  def _load_parquet(self, file, *args):
    """
    Load a parquet file from the file path created by the constructor.

    You can selectively load columns by passing an array of column names to *args.
    """
    return pq.read_pandas(os.path.join(self._path, f"{file}.parquet"), *args).to_pandas()
  #
  def save_run(self, model_run):
    """
    Save the model run

    * model_run: the number of this model run

    Runs the model, saving the results to the results folder
    """
    variant, mr_data = self.run(model_run)
    os.mkdir(f"{self._results_path}/results/model_run={model_run}")
    mr_data.to_parquet(f"{self._results_path}/results/model_run={model_run}/{model_run}.parquet")
    with open(f"{self._results_path}/selected_variants/model_run={model_run}.txt", "w") as f:
      f.write(variant)
  #
  def multi_model_runs(self, run_start, model_runs, N_CPUS = 1):
    pool = ProcessPool(ncpus = N_CPUS)
    pool.amap(self.save_run, range(run_start, run_start + model_runs))
    pool.close()
    pool.join()
  #
  def run(self, model_run):
    """
    Run the model once

    returns: a tuple of the selected varient and the updated DataFrame
    """
    pass