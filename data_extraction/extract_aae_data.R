library(tidyverse)
library(dbplyr)
library(DBI)
library(arrow)
library(janitor)

extract_aae_data <- function(providers) {
  con <- dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  tbl_aae <- tbl(con, in_schema("nhp_modelling", "aae")) %>%
    filter(fyear == 201819, activage <= 120, procode3 %in% providers) %>%
    arrange(aekey) %>%
    collect() %>%
    clean_names() %>%
    mutate(sitetret = procode3) |>
    mutate(rn = row_number(), .before = everything())
}

extract_aae_sample_data <- function() {
  con <- dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  tbl_providers_of_interest <- tbl(con, in_schema("nhp_modelling_reference", "org_code_type")) |>
    filter(org_type == "Acute", org_subtype %in% c("Small", "Medium", "Large")) |>
    select(org_code)

  tbl_aae <- tbl(con, in_schema("nhp_modelling", "aae")) %>%
    filter(fyear == 201819, activage <= 120) %>%
    semi_join(tbl_providers_of_interest, by = c("procode3" = "org_code"))

  cat("* getting n_rows: ")
  n_rows <- tbl_aae %>%
    count(procode3) %>%
    collect() %>%
    pull(n) %>%
    median() %>%
    round()
  cat(n_rows, "\n")

  cat("* getting main_icb_rate: ")
  main_icb_rate <- tbl_aae %>%
    summarise(across(is_main_icb, ~ mean(.x * 1.0, na.rm = TRUE))) %>%
    collect() %>%
    pull(is_main_icb)
  cat(main_icb_rate, "\n")

  cat("* getting aae data: ")
  # HAVE TO USE %>% rather than |>
  tbl_aae %>%
    arrange(x = NEWID()) %>%
    head(n_rows) %>%
    collect() %>%
    clean_names() %>%
    mutate(
      # create a pseudo provider field
      procode3 = "RXX",
      # make 3 sites
      sitetret = paste0("RXX0", sample(1:3, n(), TRUE)),
      is_main_icb = rbinom(n(), 1, main_icb_rate)
    ) |>
    mutate(rn = row_number(), .before = everything())
}

# ------------------------------------------------------------------------------

create_aae_data <- function(aae) {
  aae |>
    arrange(rn) |>
    select(-aekey, -hesid, -sushrg) |>
    rename(age = activage) |>
    mutate(
      across(ends_with("date"), lubridate::ymd),
      across(age, pmin, 90L),
      across(
        imd04_decile,
        fct_relevel,
        "Most deprived 10%",
        "More deprived 10-20%",
        "More deprived 20-30%",
        "More deprived 30-40%",
        "More deprived 40-50%",
        "Less deprived 40-50%",
        "Less deprived 30-40%",
        "Less deprived 20-30%",
        "Less deprived 10-20%",
        "Least deprived 10%"
      ),
      across(c(age, sex), as.integer)
    ) |>
    filter(sex %in% c(1, 2))
}

create_aae_synth_from_data <- function(data) {
  data |>
    mutate(
      across(c(age, ends_with("date")), ~ .x + sample(-5:5, n(), TRUE)),
      across(c(imd04_decile, ethnos), ~ sample(.x, n(), TRUE)),
      # "fix" age field
      age = pmin(90L, pmax(0L, age))
    )
}

aggregate_aae_data <- function(data, ...) {
  data |>
    mutate(is_ambulance = aearrivalmode == "1") |>
    select(age, sex, sitetret, is_main_icb, aedepttype, ..., matches("^(ha|i)s_")) |>
    mutate(
      hsagrp = paste(
        sep = "_",
        "aae",
        ifelse(age >= 18, "adult", "child"),
        ifelse(is_ambulance, "ambulance", "walk-in")
      )
    ) |>
    count(across(everything()), name = "arrivals") |>
    mutate(
      rn = row_number(),
      across(arrivals, as.integer),
      across(matches("^(i|ha)s\\_"), as.logical)
    ) |>
    relocate(rn, .before = everything()) |>
    drop_na()
}

save_aae_data <- function(data, name, ...) {
  path <- function(...) file.path("data", name, ...)

  if (!dir.exists(path())) {
    dir.create(path())
  }

  data |>
    arrange(age, sex, aedepttype, is_ambulance, ...) |>
    write_parquet(path("aae.parquet"))

  cat(file.size(path("aae.parquet")) / 1024^2, "\n")

  invisible(data)
}

# ------------------------------------------------------------------------------

create_synthetic_aae_extract <- function(...,
                                         name = "synthetic") {
  extract_aae_sample_data() |>
    create_aae_data() |>
    create_aae_synth_from_data() |>
    aggregate_aae_data(...) |>
    save_aae_data(name, ...)
}

create_aae_extract <- function(providers, ...,
                               name = paste(providers, collapse = "_")) {
  cat(name, ":", sep = "")
  extract_aae_data(providers) |>
    create_aae_data() |>
    aggregate_aae_data(...) |>
    save_aae_data(name, ...)
}

# ------------------------------------------------------------------------------
# create_synthetic_aae_extract(ethnos, imd04_decile)
create_synthetic_aae_extract()

purrr::walk(
  list(
    "RXC",
    "RN5",
    "RYJ",
    "RGP",
    "RNQ",
    "RD8",
    "RBZ",
    "RX1",
    "RHW",
    "RA9",
    "RGR",
    c("RXN", "RTX"),
    "RH5" # RBA" is merged in with this activity
  ),
  create_aae_extract
)
