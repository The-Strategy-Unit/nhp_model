import argparse
import os
import time

from model.AaEModel import AaEModel
from model.InpatientsModel import InpatientsModel
from model.OutpatientsModel import OutpatientsModel

def timeit(f, *args):
  """
  Time how long it takes to evaluate function `f` with arguments `*args`.
  """
  s = time.time()
  r = f(*args)
  print(f"elapsed: {time.time() - s:.3f}")
  return r

def run_model(Model, results_path, run_start, model_runs, cpus, batch_size):
  try:
    m = Model(results_path)
  except FileNotFoundError as e:
    # handle the dataset not existing: we simply skip
    if str(e).endswith(".parquet"):
      print(f"file {str(e)} not found: skipping")
    # if it's not the data file that missing, re-raise the error
    else:
      raise e
  print (f"Running: {m.__class__.__name__}")
  m.multi_model_runs(run_start, model_runs, cpus, batch_size)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("results_path", nargs = 1, help = "Path to the results")
  parser.add_argument("run_start", nargs = 1, help = "Where to start model run from", type = int)
  parser.add_argument("model_runs", nargs = 1, help = "How many model runs to perform", type = int)
  parser.add_argument("-t", "--type", default = "all", help = "Model type, either ip, op, aae, or all", type = str)
  parser.add_argument("-b", "--batch-size", default = 4, help = "Size of the batches to run the model in", type = int)
  parser.add_argument("-c", "--cpus", default = os.cpu_count(), help = "Number of CPU cores to use", type = int)
  parser.add_argument("-d", "--debug", action = "store_true")
  # Grab the Arguments
  args = parser.parse_args()
  # define the models to run
  models = { "aae": AaEModel, "ip": InpatientsModel, "op": OutpatientsModel }
  if args.type != "all":
    models = { args.type: models[args.type] }
  #
  if args.debug:
    assert args.type != "all", \
      "can only debug a single model at a time: make sure to set the --type argument"
    m = models[args.type](args.results_path[0])
    r = timeit(m.run, args.run_start[0])
    print (r)
  else:
    [
      run_model(x, args.results_path[0], args.run_start[0], args.model_runs[0], args.cpus, args.batch_size)
      for x in models.values()
    ]

if __name__ == "__main__":
  main()