import argparse
import os
import time

from model.AaEModel import AaEModel
from model.InpatientsModel import InpatientsModel
from model.OutpatientsModel import OutpatientsModel

def run_model(Model, results_path, run_start, model_runs, cpus):
  m = Model(results_path)
  print (f"Running: {m.__class__.__name__}")
  m.multi_model_runs(run_start, model_runs, cpus)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("results_path", nargs = 1, help = "Path to the results")
  parser.add_argument("run_start", nargs = 1, help = "Where to start model run from", type = int)
  parser.add_argument("model_runs", nargs = 1, help = "How many model runs to perform", type = int)
  parser.add_argument("-c", "--cpus", default = os.cpu_count(), help = "Number of CPU cores to use", type = int)
  parser.add_argument("-d", "--debug", action = "store_true")
  # Grab the Arguments
  args = parser.parse_args()
  # 
  [
    run_model(x, args.results_path[0], args.run_start[0], args.model_runs[0], args.cpus)
    for x in [AaEModel, InpatientsModel, OutpatientsModel]
  ]

if __name__ == "__main__":
  main()