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
    dplyr::rename(age = "activage") |>
    dplyr::filter(
      .data[["arrivaldate"]] >= start_date,
      .data[["arrivaldate"]] <= end_date,
      .data[["age"]] <= 120,
      .data[["sex"]] %in% c("1", "2"),
      .data[["procode3"]] %in% providers
    ) |>
    dplyr::mutate(
      is_ambulance = .data[["aearrivalmode"]] == "1",
      sitetret = .data[["procode3"]]
    ) |>
    dplyr::count(
      dplyr::across(
        c(
          "age",
          "sex",
          "sitetret",
          "is_main_icb",
          "aedepttype",
          tidyselect::matches("^(ha|i)s_")
        )
      )
    ) |>
    dplyr::collect() |>
    dplyr::mutate(
      hsagrp = paste(
        sep = "_",
        "aae",
        ifelse(.data$age >= 18, "adult", "child"),
        ifelse(.data$is_ambulance, "ambulance", "walk-in")
      ),
      dplyr::across("age", ~ pmin(.x, 90L)),
      dplyr::across(c("age", "sex"), as.integer),
      dplyr::across("n", as.integer),
      dplyr::across(tidyselect::matches("^(i|ha)s\\_"), as.logical),
      group = ifelse(.data$is_ambulance, "ambulance", "walk-in"),
      tretspef = "Other"
    ) |>
    dplyr::count(
      dplyr::across(-"n"),
      wt = .data[["n"]],
      name = "arrivals"
    ) |>
    dplyr::mutate(
      rn = dplyr::row_number(),
      .before = tidyselect::everything()
    ) |>
    janitor::clean_names() |>
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

create_provider_aae_extract <- function(params) {
  cat(paste("    [ae] running:", params$name))

  extract_aae_data(params$start_date, params$end_date, params$providers) |>
    save_aae_data(params$name, params$path)
}
