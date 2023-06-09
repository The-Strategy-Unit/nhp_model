.data <- NULL # lint helper

process_demographic_factors <- function(rds_path) {
  variant_lookup <- list(
    "principal_proj" = "principal_proj",
    "var_proj_10_year_migration" = "var_proj_10_year_migration",
    "var_proj_alt_internal_migration" = "var_proj_alt_internal_migration",
    "var_proj_high_intl_migration" = "var_proj_high_intl_migration",
    "var_proj_low_intl_migration" = "var_proj_low_intl_migration",
    "Constant fertility, no mortality improvement" = "const_fertility_no_mortality_improvement",
    "Constant fertility" = "const_fertility",
    "High population" = "high_population",
    "Young age structure" = "young_age_structure",
    "High fertility" = "high_fertility",
    "Old age structure" = "old_age_structure",
    "Low population" = "low_population",
    "Low fertility" = "low_fertility",
    "High life expectancy" = "high_life_expectancy",
    "Low life expectancy" = "low_life_expectancy",
    "No mortality improvement" = "no_mortality_improvement",
    "0% Future EU migration (Not National Statistics)" = "zero_eu_migration",
    "50% Future EU migration (Not National Statistics)" = "half_eu_migration",
    "Zero net migration (natural change only)" = "zero_net_migration",
    "Replacement fertility" = "replacement_fertility"
  )

  readRDS(rds_path) |>
    dplyr::mutate(
      dplyr::across("sex", ~ ifelse(.x == "males", 1, 2)),
      dplyr::across("age", ~ pmin(.x, 90))
    ) |>
    dtplyr::lazy_dt() |>
    dplyr::group_by(variant = .data$id, .data$sex, .data$age, .data$year, .data$procode) |>
    dplyr::summarise(
      dplyr::across("pop", ~ sum(.x, na.rm = TRUE)),
      .groups = "drop"
    ) |>
    dplyr::mutate(dplyr::across("variant", ~ as.character(variant_lookup[.]))) |>
    dplyr::as_tibble()
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

  hsa$run(trust, base_year) |> # returns filename
    stringr::str_replace_all("\\\\", "/") # returns files with \, convert to /
}
