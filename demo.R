# this is a quick sample of how to use the model results in R
# it requires you to have run the model first to create the results:
#   python data/synthetic/results/test/20220110_104353 0 30
# (that will create 30 model runs)

library(tidyverse)
library(arrow)
library(duckdb)

# load the data
ipdd <- read_parquet("data/synthetic/ip.parquet") |>
  # this is only needed during model runs, needs to be ignored in any results
  # as it isn't updated when claspat is updated
  select(-admigrp) |>
  to_duckdb()

model_results <- local({
  mr <- open_dataset("data/synthetic/results/test/20220110_104353/results") |>
    to_duckdb()

  ipdd |>
    select(-classpat, -speldur) |>
    inner_join(mr, by = "rn") |>
    # in this demo we are ignoring outpatients rows
    # though these will need to be added to the OP dataset in practice
    filter(classpat != "-1") |>
    collect() |>
    # the model only updates speldur, need to update disdate now
    mutate(disdate = admidate + speldur)
})

ip <- collect(ipdd)

# show the data
ip
model_results
