.data <- NULL # lint helper

get_demographics_factors_version <- function(demographic_factors_pin_name) {
  board <- pins::board_connect()

  pins::pin_versions(board, demographic_factors_pin_name) |>
    dplyr::filter(.data[["active"]]) |>
    _$version
}

process_demographic_factors <- function(demographic_factors_pin_name, demographic_factors_version) {
  board <- pins::board_connect()

  pop <- board |>
    pins::pin_read(demographic_factors_pin_name, demographic_factors_version) |>
    tibble::as_tibble()

  variant_lookup <- board |> # nolint
    pins::pin_read("thomas.jemmett/nhp_demographic_variants") |>
    dplyr::select("ons_id", "name") |>
    tibble::deframe()

  pop |>
    dplyr::mutate(
      dplyr::across("sex", ~ ifelse(.x == "m", 1, 2)),
      dplyr::across("age", ~ pmin(.x, 90))
    ) |>
    dtplyr::lazy_dt() |>
    dplyr::group_by(variant = .data$ons_id, .data$sex, .data$age, .data$year, .data$procode) |>
    dplyr::summarise(
      dplyr::across("pop", ~ sum(.x, na.rm = TRUE)),
      .groups = "drop"
    ) |>
    dplyr::mutate(dplyr::across("variant", ~ as.character(variant_lookup[.]))) |>
    dplyr::as_tibble() |>
    dplyr::mutate(
      dplyr::across(
        "procode",
        ~ dplyr::case_match(
          .x,
          "RH5_RBA" ~ "RH5",
          "RBZ" ~ "RH8",
          .default = .x
        )
      )
    )
}

save_synthetic_demographic_factors <- function(demographic_factors, path = "data") {
  data <- demographic_factors |>
    dplyr::select(-"procode") |>
    dplyr::group_by(dplyr::across(-"pop")) |>
    dplyr::summarise(
      dplyr::across("pop", ~ as.integer(mean(.x, na.rm = TRUE))),
      .groups = "drop"
    ) |>
    tidyr::pivot_wider(names_from = "year", values_from = "pop") |>
    dplyr::as_tibble()

  fn <- file.path(path, "synthetic", "demographic_factors.csv")
  readr::write_csv(data, fn)

  fn
}

save_demographic_factors <- function(demographic_factors, params) {
  trust <- params$name

  data <- demographic_factors |>
    dplyr::filter(.data[["procode"]] == params$providers) |>
    tidyr::pivot_wider(names_from = "year", values_from = "pop")

  fn <- file.path(params$path, trust, "demographic_factors.csv")
  readr::write_csv(data, fn)

  fn
}

create_gams <- function(org_codes, base_year = "2018") {
  trust <- paste(org_codes, collapse = "_")
  cat("      ", trust, sep = "")

  withr::local_envvar("RETICULATE_PYTHON" = "")

  reticulate::use_condaenv("nhp")
  hsa <- reticulate::import("model.hsa_gams")

  hsa$run(trust, base_year) |> # returns filename
    stringr::str_replace_all("\\\\", "/") # returns files with \, convert to /
}
