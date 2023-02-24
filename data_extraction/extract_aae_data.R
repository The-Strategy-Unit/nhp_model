.data <- NULL # lint helper
`%m+%` <- lubridate::`%m+%` # nolint
`%m-%` <- lubridate::`%m-%` # nolint

extract_aae_data <- function(start_date, end_date, providers) {
  con <- DBI::dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "aae")) |>
    dplyr::filter(
      .data$arrivaldate >= start_date,
      .data$arrivaldate <= end_date,
      .data$activage <= 120,
      .data$procode3 %in% providers
    ) |>
    dplyr::arrange(.data$aekey) |>
    dplyr::collect() |>
    janitor::clean_names() |>
    dplyr::mutate(sitetret = .data$procode3) |>
    dplyr::mutate(
      rn = dplyr::row_number(),
      .before = tidyselect::everything()
    )
}

extract_aae_sample_data <- function(start_date, end_date, ...) {
  con <- DBI::dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  tbl_providers_of_interest <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling_reference", "org_code_type")) |>
    dplyr::filter(
      .data$org_type == "Acute",
      .data$org_subtype %in% c("Small", "Medium", "Large")
    ) |>
    dplyr::select("org_code")

  tbl_aae <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "aae")) |>
    dplyr::filter(
      .data$arrivaldate >= start_date,
      .data$arrivaldate <= end_date,
      .data$activage <= 120
    ) |>
    dplyr::semi_join(tbl_providers_of_interest, by = c("procode3" = "org_code"))

  n_rows <- tbl_aae |>
    dplyr::count(.data$procode3) |>
    dplyr::collect() |>
    dplyr::pull(.data$n) |>
    median() |>
    round()

  main_icb_rate <- tbl_aae |>
    dplyr::summarise(
      dplyr::across("is_main_icb", ~ mean(.x * 1.0, na.rm = TRUE))
    ) |>
    dplyr::collect() |>
    dplyr::pull(.data$is_main_icb)

  tbl_aae |>
    dplyr::arrange(x = NEWID()) |> # nolint
    head(n_rows) |>
    dplyr::collect() |>
    janitor::clean_names() |>
    dplyr::mutate(
      # create a pseudo provider field
      procode3 = "RXX",
      # make 3 sites
      sitetret = paste0("RXX0", sample(1:3, dplyr::n(), TRUE)),
      is_main_icb = rbinom(dplyr::n(), 1, main_icb_rate)
    ) |>
    dplyr::mutate(
      rn = dplyr::row_number(),
      .before = tidyselect::everything()
    )
}

# ------------------------------------------------------------------------------

create_aae_data <- function(aae) {
  aae |>
    dplyr::arrange(.data$rn) |>
    dplyr::select(-"aekey", -"hesid", -"sushrg") |>
    dplyr::rename(age = "activage") |>
    dplyr::mutate(
      dplyr::across(tidyselect::ends_with("date"), lubridate::ymd),
      dplyr::across("age", pmin, 90L),
      dplyr::across(
        "imd04_decile",
        forcats::fct_relevel,
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
      dplyr::across(c("age", "sex"), as.integer),
      sitetret = "trust" # not currently available in our HES data
    ) |>
    dplyr::filter(.data$sex %in% c(1, 2))
}

create_aae_synth_from_data <- function(data) {
  data |>
    dplyr::mutate(
      dplyr::across(
        c("age", tidyselect::ends_with("date")),
        ~ .x + sample(-5:5, dplyr::n(), TRUE)
      ),
      dplyr::across(c("imd04_decile", "ethnos"), ~ sample(.x, dplyr::n(), TRUE)),
      # "fix" age field
      age = pmin(90L, pmax(0L, .data$age))
    )
}

aggregate_aae_data <- function(data, ...) {
  data |>
    dplyr::mutate(is_ambulance = .data$aearrivalmode == "1") |>
    dplyr::select(
      "age",
      "sex",
      "sitetret",
      "is_main_icb",
      "aedepttype",
      ...,
      tidyselect::matches("^(ha|i)s_")
    ) |>
    dplyr::mutate(
      hsagrp = paste(
        sep = "_",
        "aae",
        ifelse(.data$age >= 18, "adult", "child"),
        ifelse(.data$is_ambulance, "ambulance", "walk-in")
      )
    ) |>
    dplyr::count(
      dplyr::across(tidyselect::everything()),
      name = "arrivals"
    ) |>
    dplyr::mutate(
      rn = dplyr::row_number(),
      dplyr::across("arrivals", as.integer),
      dplyr::across(tidyselect::matches("^(i|ha)s\\_"), as.logical),
      group = ifelse(.data$is_ambulance, "ambulance", "walk-in"),
      tretspef = "Other"
    ) |>
    dplyr::relocate("rn", .before = tidyselect::everything()) |>
    tidyr::drop_na()
}

save_aae_data <- function(data, name, path, ...) {
  fn <- file.path(path, name, "aae.parquet")

  data |>
    dplyr::arrange(
      .data$age,
      .data$sex,
      .data$aedepttype,
      .data$is_ambulance,
      ...
    ) |>
    arrow::write_parquet(fn)

  fn
}

# ------------------------------------------------------------------------------

create_aae_extract <- function(start_date,
                               end_date,
                               providers,
                               name,
                               extract_fn,
                               synth_fn,
                               path,
                               ...) {
  extract_fn(start_date, end_date, providers) |>
    create_aae_data() |>
    synth_fn() |>
    aggregate_aae_data(...) |>
    save_aae_data(name, path, ...)
}

create_synthetic_aae_extract <- function(start_date,
                                         end_date,
                                         ...,
                                         name = "synthetic",
                                         path = "data") {
  create_aae_extract(
    start_date,
    end_date,
    NULL,
    name,
    extract_aae_sample_data,
    create_aae_synth_from_data,
    path,
    ...
  )
}

create_provider_aae_extract <- function(start_date,
                                        end_date,
                                        providers,
                                        ...,
                                        name,
                                        path = "data") {
  if (missing(name)) {
    name <- paste(providers, collapse = "_")
  }
  cat(paste("    running:", name))

  create_aae_extract(
    start_date,
    end_date,
    providers,
    name,
    extract_aae_data,
    identity,
    path,
    ...
  )
}
