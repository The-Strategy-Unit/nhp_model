.data <- NULL # lint helper

extract_op_data <- function(start_date, end_date, providers, specialty_fn) {
  con <- DBI::dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "outpatients")) |>
    dplyr::rename(age = "apptage") |>
    dplyr::filter(
      .data[["apptdate"]] >= start_date,
      .data[["apptdate"]] <= end_date,
      .data[["age"]] <= 120,
      .data[["sex"]] %in% c("1", "2"),
      .data[["procode3"]] %in% providers
    ) |>
    dplyr::count(
      dplyr::across(
        c(
          "age",
          "sex",
          "tretspef",
          "sitetret",
          "is_main_icb",
          tidyselect::matches("^(ha|i)s_")
        )
      )
    ) |>
    dplyr::collect() |>
    dplyr::mutate(
      type = paste(
        sep = "_",
        ifelse(.data[["is_adult"]], "adult", "child"),
        ifelse(.data[["is_surgical_specialty"]], "surgical", "non-surgical")
      ),
      dplyr::across(tidyselect::matches("^(i|ha)s\\_"), as.logical),
      dplyr::across("has_procedures", \(.x) .x & !.data[["is_tele_appointment"]]),
      group = dplyr::case_when(
        .data[["has_procedures"]] ~ "procedure",
        .data[["is_first"]] ~ "first",
        .default = "followup"
      ),
      hsagrp = paste(
        sep = "_",
        "op",
        .data[["type"]],
        .data[["group"]]
      ),
      dplyr::across(tidyselect::ends_with("date"), lubridate::ymd),
      dplyr::across("age", ~ pmin(.x, 90L)),
      dplyr::across(c("age", "sex"), as.integer),
      tretspef_raw = .data[["trespef"]],
      dplyr::across("tretspef", specialty_fn),
      is_wla = TRUE
    ) |>
    dplyr::group_by(
      dplyr::across(c(tidyselect::everything(), -"is_tele_appointment"))
    ) |>
    dplyr::summarise(
      attendances = sum((!.data[["is_tele_appointment"]]) * .data[["n"]]),
      tele_attendances = sum(.data[["is_tele_appointment"]] * .data[["n"]]),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      rn = dplyr::row_number(), .before = tidyselect::everything()
    ) |>
    janitor::clean_names() |>
    tidyr::drop_na()
}

# ------------------------------------------------------------------------------

save_op_data <- function(data, name, path, ...) {
  fn <- file.path(path, name, "op.parquet")

  data |>
    dplyr::arrange(
      .data$age,
      .data$sex,
      .data$tretspef,
      .data$is_main_icb,
      ...
    ) |>
    arrow::write_parquet(fn)

  fn
}

# ------------------------------------------------------------------------------

create_provider_op_extract <- function(params, specialties = NULL) {
  cat(paste("    [op] running:", params$name))

  specialty_fn <- if (is.null(specialties)) {
    identity
  } else {
    function(.x) {
      dplyr::case_when(
        .x %in% specialties ~ .x,
        stringr::str_detect(.x, "^1(?!80|9[02])") ~
          "Other (Surgical)",
        stringr::str_detect(.x, "^(1(80|9[02])|[2346]|5(?!60)|83[134])") ~
          "Other (Medical)",
        TRUE ~ "Other"
      )
    }
  }

  extract_op_data(
    params$start_date,
    params$end_date,
    params$providers,
    specialty_fn
  ) |>
    save_op_data(params$name, params$path)
}
