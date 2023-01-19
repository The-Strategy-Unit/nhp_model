.data <- NULL # lint helper

extract_op_data <- function(start_date, end_date, providers) {
  con <- DBI::dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "outpatients")) |>
    dplyr::filter(
      .data$apptdate >= start_date,
      .data$apptdate <= end_date,
      .data$apptage <= 120,
      .data$procode3 %in% providers
    ) |>
    dplyr::collect() |>
    janitor::clean_names() |>
    dplyr::arrange(.data$attendkey) |>
    dplyr::mutate(
      rn = dplyr::row_number(),
      .before = tidyselect::everything()
    )
}

extract_op_sample_data <- function(start_date, end_date, ...) {
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

  tbl_outpatients <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "outpatients")) |>
    dplyr::filter(
      .data$apptdate >= start_date,
      .data$apptdate <= end_date,
      .data$apptage <= 120
    ) |>
    dplyr::semi_join(tbl_providers_of_interest, by = c("procode3" = "org_code"))

  n_rows <- tbl_outpatients |>
    dplyr::count(.data$procode3) |>
    dplyr::collect() |>
    dplyr::pull(.data$n) |>
    median() |>
    round()

  main_icb_rate <- tbl_outpatients |>
    dplyr::summarise(
      dplyr::across("is_main_icb", ~ mean(.x * 1.0, na.rm = TRUE))
    ) |>
    dplyr::collect() |>
    dplyr::pull(.data$is_main_icb)

  outpatients <- tbl_outpatients |>
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
    mutate(
      rn = dplyr::row_number(),
      .before = tidyselect::everything()
    )

  outpatients
}

# ------------------------------------------------------------------------------

create_op_data <- function(outpatients, specialties = NULL) {
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

  outpatients |>
    dplyr::arrange(.data$rn) |>
    dplyr::select(-"fyear", -"attendkey", -"encrypted_hesid", -"firstatt") |>
    dplyr::rename(age = "apptage") |>
    dplyr::mutate(
      dplyr::across(tidyselect::ends_with("date"), lubridate::ymd),
      dplyr::across(tidyselect::ends_with("age"), pmin, 90),
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
      dplyr::across("tretspef", specialty_fn),
      dplyr::across("has_procedures", `*`, (1 - .data$is_tele_appointment)),
      dplyr::across(tidyselect::matches("^(i|ha)s\\_"), as.logical)
    ) |>
    dplyr::filter(.data$sex %in% c(1, 2))
}

create_op_synth_from_data <- function(data) {
  hrg_by_tretspef <- data |>
    dplyr::group_by(
      .data$tretspef,
      .data$is_adult,
      .data$is_first,
      .data$is_tele_appointment,
      .data$has_procedures
    ) |>
    dplyr::count(.data$sushrg, name = "sushrg_n") |>
    dplyr::summarise(
      dplyr::across(tidyselect::everything(), list),
      .groups = "drop"
    )

  data |>
    dplyr::mutate(
      dplyr::across(c("age", "apptdate"), ~ .x + sample(-5:5, dplyr::n(), TRUE)),
      dplyr::across(c("imd04_decile", "ethnos"), ~ sample(.x, dplyr::n(), TRUE)),
      age = dplyr::case_when(
        .data$age < 0L ~ 0L,
        .data$age > 90L ~ 90L,
        TRUE ~ .data$age
      ),
      is_adult = as.integer(.data$age >= 18),
      is_surgical_specialty = as.integer(
        stringr::str_detect(.data$tretspef, "^1(?!80|9[012])")
      )
    ) |>
    dplyr::group_by(
      is_0_yo = age == 0,
      dplyr::across(tidyselect::matches("^(i|ha)\\_"))
    ) |>
    dplyr::mutate(
      dplyr::across("tretspef", ~ sample(.x, dplyr::n(), TRUE))
    ) |>
    dplyr::select(-"sushrg") |>
    # make sure to shuffle hrg's only within an acceptible list from that
    # specialty
    dplyr::inner_join(hrg_by_tretspef) |>
    mutate(
      dplyr::across("sushrg", purrr::map2_chr, .data$sushrg_n, sample, size = 1, replace = FALSE)
    ) |>
    dplyr::ungroup() |>
    dplyr::select(-"sushrg_n", -"is_0_yo") |>
    dplyr::relocate("sushrg", .after = "refsourc")
}

aggregate_op_data <- function(data, ...) {
  data |>
    dplyr::select("age", "sex", "tretspef", "sitetret", "is_main_icb", ..., tidyselect::matches("^(ha|i)s_")) |>
    dplyr::mutate(
      type = paste(
        sep = "_",
        ifelse(.data$is_adult, "adult", "child"),
        ifelse(.data$is_surgical_specialty, "surgical", "non-surgical")
      ),
      hsagrp = paste(
        sep = "_",
        "op",
        .data$type,
        dplyr::case_when(
          .data$has_procedures ~ "procedure",
          .data$is_first ~ "first",
          TRUE ~ "follow-up"
        )
      )
    ) |>
    dplyr::group_by(
      dplyr::across(c(tidyselect::everything(), -"is_tele_appointment"))
    ) |>
    dplyr::summarise(
      attendances = sum(1 - .data$is_tele_appointment, na.rm = TRUE),
      tele_attendances = sum(.data$is_tele_appointment, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      rn = dplyr::row_number(),
      dplyr::across("attendances", as.integer),
      dplyr::across(tidyselect::matches("^(i|ha)s\\_"), as.logical)
    ) |>
    dplyr::relocate("rn", .before = everything()) |>
    tidyr::drop_na()
}

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

create_op_extract <- function(start_date,
                              end_date,
                              providers,
                              specialties,
                              name,
                              extract_fn,
                              synth_fn,
                              path,
                              ...) {
  extract_fn(start_date, end_date, providers) |>
    create_op_data(specialties) |>
    synth_fn() |>
    aggregate_op_data(...) |>
    save_op_data(name, path, ...)
}

create_synthetic_op_extract <- function(start_date,
                                        end_date,
                                        ...,
                                        name = "synthetic",
                                        specialties = NULL,
                                        path = "data") {
  create_op_extract(
    start_date,
    end_date,
    NULL,
    specialties,
    name,
    extract_op_sample_data,
    create_op_synth_from_data,
    path,
    ...
  )
}

create_provider_op_extract <- function(start_date,
                                       end_date,
                                       providers,
                                       ...,
                                       name,
                                       specialties = NULL,
                                       path = "data") {
  if (missing(name)) {
    name <- paste(providers, collapse = "_")
  }
  cat(paste("    running:", name))

  create_op_extract(
    start_date,
    end_date,
    providers,
    specialties,
    name,
    extract_op_data,
    identity,
    path,
    ...
  )
}
