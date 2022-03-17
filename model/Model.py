import json
import pickle
import os

from time import time

import numpy as np
import pyarrow.parquet as pq
import pandas as pd

from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

from model.helpers import rnorm

class Model:
  """
  Inpatients Model
  
  * results_path: where the data is stored

  This is a generic implementation of the model. Specific implementations of the model inherit from this class.

  In order to run the model you need to pass the path to where the parameters json file is stored. This path should be
  of the form: {dataset_name}/{results}/{model_name}/{model_run_datetime}
  
  Once the object is constructed you can call either m.run() to run the model and return the data, or
  m.save_run() to run the model and save the results.

  You can also call m.multi_model_run() to run multiple iterations of the model in parallel, saving the results to disk
  to use later.
  """
  def __init__(self, results_path, columns_to_load = None):
    # load the parameters file
    with open(f"{results_path}/params.json", "r") as f: self._params = json.load(f)
    # store the path where the data is stored and the results are stored
    self._path = str(Path(results_path).parent.parent.parent)
    self._results_path = results_path
    #
    # load the data that's shared across different model types
    #
    with open(f"{self._path}/hsa_gams.pkl", "rb") as f: self._hsa_gams = pickle.load(f)
    self._load_demog_factors()# load the data. we only need some of the columns for the model, so just load what we need
    data = (self
      ._load_parquet(self._MODEL_TYPE, columns_to_load)
      # merge the demographic factors to the data
      .merge(self._demog_factors, left_on = ["age", "sex"], right_index = True)
      .groupby(["variant"])
    )
    # we now store the data in a dictionary keyed by the population variant
    self._data = { k: v.drop(["variant"], axis = "columns").set_index(["rn"]) for k, v in tuple(data) }
  #
  def _select_variant(self, rng):
    """
    Randomly select a single variant to use for a model run
    """
    v = rng.choice(self._variants, p = self._probabilities)
    return (v, self._data[v])
  #
  def _load_demog_factors(self):
    dfp = self._params["demographic_factors"]
    start_year = dfp.get("start_year", "2018")
    end_year = dfp.get("end_year", "2043")
    #
    df = pd.read_csv(os.path.join(self._path, dfp["file"]))
    df[["age", "sex"]] = df[["age", "sex"]].astype(int)
    df["factor"] = df[end_year] / df[start_year]
    self._demog_factors = df.set_index(["age", "sex"])[["variant", "factor"]]
    #
    self._variants = list(dfp["variant_probabilities"].keys())
    self._probabilities = list(dfp["variant_probabilities"].values())
  #
  def _health_status_adjustment(self, rng, data):
    params = self._params["health_status_adjustment"]
    #
    ages = np.arange(params["min_age"], params["max_age"] + 1)
    adjusted_ages = ages - [rnorm(rng, *i) for i in params["intervals"]]
    hsa = pd.concat([
      pd.DataFrame({
        "hsagrp": a,
        "sex": int(s),
        "age": ages,
        "hsa_f": g.predict(adjusted_ages) / g.predict(ages)
      })
      for (a, s), g in self._hsa_gams.items()
    ])
    data = (data
      .reset_index()
      .merge(hsa, on = ["hsagrp", "sex", "age"], how = "left")
      .set_index(["rn"])
    )
    # because we do a left join, some groups / sex / age rows may be NaN. replace with multiplication identity (1)
    data["hsa_f"].fillna(1, inplace = True)
    #
    return data
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
    Save the model run and run parameters

    * model_run: the number of this model run

    returns: a tuple containing the time to run the model, and the time to save the results
    """
    t0 = time()
    params, mr_data = self.run(model_run)
    t1 = time()
    #
    results_path = f"{self._results_path}/model_run={model_run}"
    if not os.path.exists(results_path): os.makedirs(results_path)
    #
    mr_data.to_parquet(f"{results_path}/{self._MODEL_TYPE}.parquet")
    with open(f"{results_path}/{self._MODEL_TYPE}_params.json", "w") as f:
      json.dump(params, f)
    #
    t2 = time()
    return (t1 - t0, t2 - t1)
  #
  def multi_model_runs(self, run_start, model_runs, N_CPUS = 1):
    print (f"{model_runs} model runs on {N_CPUS} cpus")
    pbar = tqdm(total = model_runs)
    with Pool(N_CPUS) as pool:
      results = [
        pool.apply_async(
          self.save_run,
          (i, ),
          callback = lambda _: pbar.update()
        )
        for i in range(run_start, run_start + model_runs)
      ]
      for r in results: r.wait()
    #
    times = [r.get() for r in results]
    run_times = [t[0] for t in times]
    save_times = [t[1] for t in times]
    total_times = [t[0] + t[1] for t in times]
    display_times = lambda t: f"mean: {np.mean(t):.3f}, range: [{np.min(t):.3f}, {np.max(t):.3f}]"
    print("\nTimings:")
    print("=" * 80)
    print(f"* model runs:    {display_times(run_times)}")
    print(f"* save results:  {display_times(save_times)}")
    print(f"* total elapsed: {display_times(total_times)}")
    print("=" * 80)
  #
  def run(self, model_run):
    """
    Run the model once

    returns: a tuple of the selected varient and the updated DataFrame
    """
    pass