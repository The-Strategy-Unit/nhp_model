# this is a quick sample of how to use the model results in R
# it requires you to have run the model first to create the results.
# see run_model.ipynb

library(tidyverse)
library(arrow)
library(duckdb)

# load the aggregated results
# ---------------------------
# * model_run = -1 is the baseline data, and as such has no population "variant"
# * model_run = 0 is the "principal" model run
# * model_run = 1..n are the monte carlo simulation model runs
model_results <- local({
  mr <- open_dataset("results/aggregated_results") |>
    collect()

  selected_variants <- mr |>
    distinct(dataset, scenario, create_datetime) |>
    mutate(variant = pmap(
      list(dataset, scenario, create_datetime),
      \(ds, sc, cd) {
        fn <- glue::glue(
          "results/run_params/dataset={ds}/scenario={sc}/{cd}.json"
        )
        jsonlite::read_json(fn, simplifyVector = TRUE)$variant |>
          enframe(name = "model_run", value = "variant") |>
          mutate(across(model_run, `-`, 1))
      }
    )) |>
    unnest(variant)

  mr |>
    left_join(
      selected_variants,
      by = c("dataset", "scenario", "create_datetime", "model_run")
    )
})

# load the change factors
# -----------------------
# there will be no rows with model_run = -1 here as this is just showing the
# results of the individual model steps
# the change_factor rows should largely be treated in order of appearance:
# * the "baseline" row must come first
# * the rest of the change factors appear in the order that they are executed
#   within the model
#
# the "measure" column will contain the figure that was adjusted in the model
# step. For example, in inpatients we have "admissions" and "beddays".
change_factors <- open_dataset("results/change_factors", format = "csv") |>
  collect()

change_factors