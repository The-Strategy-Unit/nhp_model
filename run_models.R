library(nhpmodel)

task_jsons <- commandArgs(trailingOnly = TRUE)

stopifnot(
  "some/all provided files do not exist" = purrr::every(task_jsons, file.exists)
)

options(
  nhp_model_data_path = Sys.getenv("DATA_PATH", "/mnt/data"),
  nhp_model_queue_path = Sys.getenv("QUEUE_PATH", "/mnt/queue"),
  nhp_model_results_path = Sys.getenv("RESULTS_PATH", "/mnt/results"),
  nhp_model_ncpus = as.integer(Sys.getenv("NCPUS", future::availableCores()))
)

cat ("running with", getOption("nhp_model_ncpus"), "cpus\n")

purrr::walk(task_jsons, run_model)

cat("\n")