library(tidyverse)
library(dbplyr)
library(DBI)
library(arrow)
library(janitor)

extract_op_data <- function(providers) {
  con <- dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  tbl(con, in_schema("nhp_modelling", "outpatients")) %>%
    filter(fyear == 201819, apptage <= 120, procode3 %in% providers) %>%
    collect() %>%
    clean_names() %>%
    arrange(attendkey) %>%
    mutate(rn = row_number(), .before = everything())
}

extract_op_sample_data <- function() {
  con <- dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  tbl_providers_of_interest <- tbl(con, in_schema("nhp_modelling_reference", "org_code_type")) |>
    filter(org_type == "Acute", org_subtype %in% c("Small", "Medium", "Large")) |>
    select(org_code)

  tbl_outpatients <- tbl(con, in_schema("nhp_modelling", "outpatients")) %>%
    filter(fyear == 201819, apptage <= 120) %>%
    semi_join(tbl_providers_of_interest, by = c("procode3" = "org_code"))

  cat("* getting n_rows: ")
  n_rows <- tbl_outpatients %>%
    count(procode3) %>%
    collect() %>%
    pull(n) %>%
    median() %>%
    round()
  cat(n_rows, "\n")

  cat("* getting main_icb_rate: ")
  main_icb_rate <- tbl_outpatients %>%
    summarise(across(is_main_icb, ~ mean(.x * 1.0, na.rm = TRUE))) %>%
    collect() %>%
    pull(is_main_icb)
  cat(main_icb_rate, "\n")

  cat("* getting outpatients data: ")
  outpatients <- tbl_outpatients %>%
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
  cat("done\n")

  outpatients
}

# ------------------------------------------------------------------------------

create_op_data <- function(outpatients, specialties = NULL) {
  specialty_fn <- if (is.null(specialties)) {
    identity
  } else {
    function(.x) {
      case_when(
        .x %in% specialties ~ .x,
        str_detect(.x, "^1(?!80|9[02])") ~
          "Other (Surgical)",
        str_detect(.x, "^(1(80|9[02])|[2346]|5(?!60)|83[134])") ~
          "Other (Medical)",
        TRUE ~ "Other"
      )
    }
  }

  outpatients |>
    arrange(rn) |>
    select(-fyear, -attendkey, -encrypted_hesid, -firstatt) |>
    rename(age = apptage) |>
    mutate(
      across(ends_with("date"), lubridate::ymd),
      across(ends_with("age"), pmin, 90),
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
      across(c(age, sex), as.integer),
      across(tretspef, specialty_fn),
      across(has_procedures, `*`, (1 - is_tele_appointment)),
      across(matches("^(i|ha)s\\_"), as.logical)
    ) |>
    filter(sex %in% c(1, 2))
}

create_op_synth_from_data <- function(data) {
  hrg_by_tretspef <- data |>
    group_by(
      tretspef,
      is_adult,
      is_first,
      is_tele_appointment,
      has_procedures
    ) |>
    count(sushrg, name = "sushrg_n") |>
    summarise(across(everything(), list), .groups = "drop")

  data |>
    mutate(
      across(c(age, apptdate), ~ .x + sample(-5:5, n(), TRUE)),
      across(c(imd04_decile, ethnos), ~ sample(.x, n(), TRUE)),
      age = case_when(
        age < 0L ~ 0L,
        age > 90L ~ 90L,
        TRUE ~ age
      ),
      is_adult = as.integer(age >= 18),
      is_surgical_specialty = as.integer(
        str_detect(tretspef, "^1(?!80|9[012])")
      )
    ) |>
    group_by(is_0_yo = age == 0, across(matches("^(i|ha)\\_"))) |>
    mutate(across(tretspef, ~ sample(.x, n(), TRUE))) |>
    select(-sushrg) |>
    # make sure to shuffle hrg's only within an acceptible list from that
    # specialty
    inner_join(hrg_by_tretspef) |>
    mutate(
      across(sushrg, map2_chr, sushrg_n, sample, size = 1, replace = FALSE)
    ) |>
    ungroup() |>
    select(-sushrg_n, -is_0_yo) |>
    relocate(sushrg, .after = refsourc)
}

aggregate_op_data <- function(data, ...) {
  data |>
    select(age, sex, tretspef, sitetret, is_main_icb, ..., matches("^(ha|i)s_")) |>
    mutate(
      type = paste(
        sep = "_",
        ifelse(is_adult, "adult", "child"),
        ifelse(is_surgical_specialty, "surgical", "non-surgical")
      ),
      hsagrp = paste(
        sep = "_",
        "op",
        type,
        case_when(
          has_procedures ~ "procedure",
          is_first ~ "first",
          TRUE ~ "follow-up"
        )
      )
    ) |>
    group_by(across(c(everything(), -is_tele_appointment))) |>
    summarise(
      attendances = sum(1 - is_tele_appointment, na.rm = TRUE),
      tele_attendances = sum(is_tele_appointment, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      rn = row_number(),
      across(attendances, as.integer),
      across(matches("^(i|ha)s\\_"), as.logical)
    ) |>
    relocate(rn, .before = everything()) |>
    drop_na()
}

save_op_data <- function(data, name, ...) {
  path <- function(...) file.path("data", name, ...)

  if (!dir.exists(path())) {
    dir.create(path())
  }

  data |>
    arrange(age, sex, tretspef, is_main_icb, ...) |>
    write_parquet(path("op.parquet"))

  cat(file.size(path("op.parquet")) / 1024^2, "\n")

  invisible(data)
}

# ------------------------------------------------------------------------------

create_synthetic_op_extract <- function(...,
                                        name = "synthetic",
                                        specialties = NULL) {
  extract_op_sample_data() |>
    create_op_data(specialties) |>
    create_op_synth_from_data() |>
    aggregate_op_data(...) |>
    save_op_data(name, ...)
}

create_provider_op_extract <- function(providers,
                                       ...,
                                       name = paste(providers, collapse = "_"),
                                       specialties = NULL) {
  cat(name, ":", sep = "")
  extract_op_data(providers) |>
    create_op_data(specialties) |>
    aggregate_op_data(...) |>
    save_op_data(name, ...)
}

# ------------------------------------------------------------------------------

rtt_specs <- c(
  "100", "101", "110", "120", "130", "140", "160", "300", "320", "330", "400",
  "410", "430", "502"
)

# create_synthetic_op_extract(ethnos, imd04_decile, specialties = rtt_specs)
create_synthetic_op_extract(specialties = rtt_specs)

purrr::walk(
  list(
    "RA9",
    "RD8",
    "RGP",
    "RGR",
    "RH5", # "RBA" is merged in with this activity
    "RH8", # was "RBZ",
    "RHW",
    "RN5",
    "RNQ",
    "RX1",
    "RXC",
    c("RXN", "RTX"),
    "RYJ"
  ),
  create_provider_op_extract,
  specialties = rtt_specs
)
