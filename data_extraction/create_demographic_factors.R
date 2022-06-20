library(tidyverse)
library(dtplyr)
# load reticulate, but make sure to handle case of using radian
Sys.setenv("RETICULATE_PYTHON" = "")
library(reticulate)
use_condaenv("nhp")
hsa <- import("model.hsa_gams")

rds_path <- file.path("_scratch", "demographic_factors.rds")

df <- local({
  df <- readRDS(rds_path) |>
    mutate(
      across(sex, ~ ifelse(.x == "males", 1, 2)),
      across(age, pmin, 90)
    ) |>
    arrange(id, sex, age, year, procode) |>
    lazy_dt() |>
    group_by(variant = id, sex, age, year, procode) |>
    summarise(across(pop, sum, na.rm = TRUE), .groups = "drop_last")

  df_synth <- df |>
    summarise(across(pop, compose(as.integer, mean), na.rm = TRUE), .groups = "drop") |>
    as_tibble() |>
    mutate(procode = "synthetic")

  df |>
    as_tibble() |>
    bind_rows(df_synth)
})

df |>
  pivot_wider(names_from = year, values_from = pop) |>
  group_nest(procode) |>
  transmute(
    file = file.path("data", procode, "demographic_factors.csv"),
    x = data
  ) |>
  pwalk(write_csv)

# create gams
df |>
  pluck("procode") |>
  unique() |>
  walk(hsa$create_gams, "2020")
