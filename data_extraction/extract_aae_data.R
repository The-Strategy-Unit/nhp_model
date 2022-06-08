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
    mutate(rn = row_number(), .before = everything())
}

extract_aae_sample_data <- function() {
  con <- dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  # only select organisations that had on average at least 50 people attendances
  # per day
  providers_of_interest <- tbl(con, sql("
    SELECT
      t.procode3
    FROM (
      SELECT DISTINCT
        a.procode3,
        AVG(COUNT(*)) OVER(PARTITION BY a.procode3) attendances
      FROM
        nhp_modelling.aae a
      WHERE
        a.fyear = 201819
      AND
        a.procode3 LIKE 'R%'
      GROUP BY
        a.procode3,
        a.arrivaldate
      HAVING
      -- make sure the mean age is over 18, should exclude Children's hospitals
        AVG(activage) > 18
    ) t
    WHERE
      t.attendances > 200
  "))
  tbl_providers_of_interest <- copy_to(
    con, providers_of_interest, "providers_of_interest"
  )

  # in case we need to view which providers are selected
  # ods <- tbl(con, sql("SELECT [Organisation_Code], [Organisation_Name]
  #                      FROM   [UK_Health_Dimensions].[ODS].[NHS_Trusts_SCD]
  #                      WHERE  [Is_Latest] = 1")) %>%
  #   collect()
  #
  # collect(tbl_providers_of_interest) %>%
  #   left_join(ods, by = c("PROCODE3" = "Organisation_Code")) %>%
  #   arrange(Organisation_Name) %>%
  #   View()

  tbl_aae <- tbl(con, in_schema("nhp_modelling", "aae")) %>%
    filter(fyear == 201819, activage <= 120) %>%
    semi_join(tbl_providers_of_interest, by = "procode3")

  n_rows <- tbl_aae %>%
    count(procode3) %>%
    collect() %>%
    pull(n) %>%
    median() %>%
    round()

  # HAVE TO USE %>% rather than |>
  tbl_aae %>%
    arrange(x = NEWID()) %>%
    head(n_rows) %>%
    collect() %>%
    clean_names() %>%
    mutate(rn = row_number(), .before = everything())
}

# ------------------------------------------------------------------------------

create_aae_data <- function(aae) {
  aae |>
    arrange(rn) |>
    select(-aekey, -hesid, -procode5, -sushrg) |>
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
    select(age, sex, aedepttype, aearrivalmode, ..., matches("^(ha|i)s_")) |>
    mutate(
      hsagrp = paste(
        sep = "_",
        "aae",
        ifelse(age >= 18, "adult", "child"),
        ifelse(aearrivalmode == 1, "ambulance", "walk-in")
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
  path <- function(...) file.path("../../nhp_model/data/", name, ...)

  if (!dir.exists(path())) {
    dir.create(path())
  }

  data |>
    arrange(age, sex, aedepttype, aearrivalmode, ...) |>
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
# create_synthetattendkey  ic_aae_extract()

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
    c("RH5", "RBA")
  ),
  create_aae_extract
)
