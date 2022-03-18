import argparse
import os
import time

from model.InpatientsModel import InpatientsModel

def timeit(f, *args):
  """
  Time how long it takes to evaluate function `f` with arguments `*args`.
  """
  s = time.time()
  r = f(*args)
  print(f"elapsed: {time.time() - s:.3f}")
  return r

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("results_path", nargs = 1, help = "Path to the results")
  parser.add_argument("run_start", nargs = 1, help = "Where to start model run from", type = int)
  parser.add_argument("model_runs", nargs = 1, help = "How many model runs to perform", type = int)
  parser.add_argument("-c", "--cpus", default = os.cpu_count(), help = "Number of CPU cores to use", type = int)
  parser.add_argument("-d", "--debug", action = "store_true")
  # Grab the Arguments
  args = parser.parse_args()
  # run the model
  m = InpatientsModel(args.results_path[0])
  if args.debug:
    _, r = timeit(m.run, 0)
    print (r)
  else:
    m.multi_model_runs(args.run_start[0], args.model_runs[0], args.cpus)

if __name__ == "__main__":
  main()