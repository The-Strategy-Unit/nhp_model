generate_aae_from_ecds <- function(params, ecds_extract, successors_file) {
  force(params)

  successors <- readr::read_csv(successors_file, col_types = "cc")

  df <- arrow::read_parquet(ecds_extract) |>
    dplyr::filter(.data[["attendance_category"]] == "1") |>
    tibble::as_tibble() |>
    dplyr::inner_join(successors, by = dplyr::join_by("procode" == "old_code")) |>
    dplyr::mutate(
      procode = dplyr::case_when(
        .data[["sitetret"]] %in% c("RW602", "RM318") ~ "R0A",
        .default = .data[["new_code"]]
      )
    ) |>
    dplyr::select(-"new_code", -"rn") |>
    dplyr::count(
      dplyr::across(-"arrivals"),
      wt = .data[["arrivals"]],
      name = "arrivals"
    ) |>
    dplyr::mutate(
      rn = dplyr::row_number(),
      .by = "procode",
      .before = tidyselect::everything()
    )

  params <- params |>
    tibble::enframe() |>
    tidyr::separate("name", c("x", "y", "z"), "_", extra = "merge") |>
    tidyr::pivot_wider(names_from = "z", values_from = "value") |>
    dplyr::select(-"x", -"y") |>
    tidyr::unnest(tidyselect::everything()) |>
    dplyr::mutate(year = lubridate::year(.data[["start_date"]])) |>
    dplyr::select(procode = "name", "year", "path")

  df |>
    dplyr::group_nest(.data[["procode"]], .data[["fyear"]]) |>
    dplyr::mutate(year = as.numeric(stringr::str_sub(.data[["fyear"]], 1, 4))) |>
    dplyr::inner_join(params, by = dplyr::join_by("procode", "year")) |>
    purrr::pmap(\(procode, data, path, ...) {
      p <- glue::glue("{path}/{procode}")

      if (!dir.exists(p)) {
        return()
      }

      fn <- glue::glue("{p}/aae.parquet")
      arrow::write_parquet(data, fn)
      fn
    }) |>
    purrr::flatten_chr()
}

generate_ecds_inputs_data <- function() {
  cut_age <- function(age, break_size = 5, upper_age = 90) {
    breaks <- seq(0, upper_age, break_size)

    label_fn <- function(.x) {
      if (.x == upper_age) {
        return(paste0(upper_age, "+"))
      }
      .x <- stringr::str_pad(.x + c(0, break_size - 1), 2, pad = "0")
      paste(.x, collapse = "-")
    }

    labels <- purrr::map_chr(breaks, label_fn)

    cut(age, c(breaks, Inf), labels, right = FALSE) |>
      as.character()
  }

  df |>
    dplyr::mutate(
      fyear = 201920,
      is_adult = .data$age >= 18,
      age_group = cut_age(.data[["age"]])
    ) |>
    dplyr::group_by(
      .data[["fyear"]],
      procode3 = .data[["procode"]],
      .data[["age_group"]],
      .data[["sex"]],
      .data[["is_ambulance"]],
      .data[["is_adult"]]
    ) |>
    dplyr::summarise(
      dplyr::across(
        c(
          "is_low_cost_referred_or_discharged",
          "is_left_before_treatment",
          "is_frequent_attender",
          "is_discharged_no_treatment"
        ),
        \(.x) sum(.x, na.rm = TRUE)
      ),
      n = dplyr::n(),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      subgroup = paste(
        ifelse(.data$is_adult, "adult", "child"),
        ifelse(.data$is_ambulance, "ambulance", "walk-in"),
        sep = "_"
      )
    ) |>
    dplyr::select(-"is_adult", -"is_ambulance") |>
    dplyr::rename(
      "low_cost_discharged" = "is_low_cost_referred_or_discharged",
      "left_before_seen" = "is_left_before_treatment",
      "frequent_attenders" = "is_frequent_attender",
      "discharged_no_treatment" = "is_discharged_no_treatment"
    ) |>
    tidyr::pivot_longer(
      c(
        "low_cost_discharged",
        "left_before_seen",
        "frequent_attenders",
        "discharged_no_treatment"
      )
    ) |>
    dplyr::arrange(.data$age_group) |>
    dplyr::transmute(
      .data$fyear,
      .data$procode3,
      dplyr::across("age_group", forcats::fct_inorder),
      .data$sex,
      strategy = glue::glue("{.data$name}_{.data$subgroup}"),
      .data$value,
      .data$n
    )
}
