.data <- NULL # lint helper

process_demographic_factors <- function(rds_path) {
  readRDS(rds_path) |>
    dplyr::mutate(
      dplyr::across("sex", ~ ifelse(.x == "males", 1, 2)),
      dplyr::across("age", pmin, 90)
    ) |>
    dtplyr::lazy_dt() |>
    dplyr::group_by(variant = .data$id, .data$sex, .data$age, .data$year, .data$procode) |>
    dplyr::summarise(
      dplyr::across("pop", sum, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::as_tibble()
}

save_synthetic_demographic_factors <- function(demographic_factors, path = "data") {
  data <- demographic_factors |>
    dplyr::select(-"procode") |>
    dplyr::group_by(dplyr::across(-"pop")) |>
    dplyr::summarise(
      dplyr::across("pop", purrr::compose(as.integer, mean), na.rm = TRUE),
      .groups = "drop"
    ) |>
    tidyr::pivot_wider(names_from = "year", values_from = "pop") |>
    dplyr::as_tibble()

  fn <- file.path(path, "synthetic", "demographic_factors.csv")
  readr::write_csv(data, fn)

  fn
}

save_demographic_factors <- function(demographic_factors, org_codes, path = "data") {
  trust <- paste(org_codes, collapse = "_")

  data <- demographic_factors |>
    dplyr::filter(.data$procode == trust) |>
    tidyr::pivot_wider(names_from = "year", values_from = "pop")

  fn <- file.path(path, trust, "demographic_factors.csv")
  readr::write_csv(data, fn)

  fn
}

create_gams <- function(org_codes, base_year = "2018") {
  trust <- paste(org_codes, collapse = "_")
  cat("      ", trust, sep = "")

  withr::local_envvar("RETICULATE_PYTHON" = "")

  reticulate::use_condaenv("nhp")
  hsa <- reticulate::import("model.hsa_gams")

  hsa$run(trust, "2018") |> # returns filename
    stringr::str_replace_all("\\\\", "/") # returns files with \, convert to /
}
