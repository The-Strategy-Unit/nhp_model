.data <- NULL # lint helper
`%m+%` <- lubridate::`%m+%` # nolint -- import function from luibridate
where <- NULL # lint helper

theatres_get_four_hour_sessions <- function(theatres_data_path, start_date = lubridate::ymd(20190401)) {
  `%m+%` <- lubridate::`%m+%` # nolint -- import function from luibridate
  readr::read_csv(theatres_data_path, col_types = "cccdddd") |>
    dplyr::mutate(
      # for some reason read_csv isn't able to parse the dates...
      dplyr::across("reporting_date", lubridate::dmy)
    ) |>
    dplyr::inner_join(
      tibble::tribble(
        ~sub_specialty, ~specialty_code,
        "General Surgery", "100",
        "Urology", "101",
        "TandO", "110",
        "Ear Nose and Throat", "120",
        "Ophthalmology", "130",
        "Oral and Maxfax", "140",
        "Plastics", "160",
        "Obs and Gynae", "502",
        "Trust", "Other (Surgical)"
      ),
      by = "sub_specialty"
    ) |>
    dplyr::filter(
      .data$reporting_date >= start_date,
      .data$reporting_date < start_date %m+% lubridate::years(1)
    ) |>
    dplyr::group_by(.data$org_code, .data$specialty_code) |>
    dplyr::summarise(
      dplyr::across("four_hour_sessions", sum),
      .groups = "drop_last"
    ) |>
    dplyr::mutate(
      dplyr::across(
        "four_hour_sessions",
        # the "Other" row was the trust total. using the following we will end up with a column where the sum
        # is equal to the trusts total.
        #   T = sum(X), (total is the sum of all the values)
        #   S = T + sum(x), (this column we have the total + sum of a subset of X)
        #   T = sum(x) + X_o, (T is sum of our subset + the other value from X)
        #   S = 2T - X_o  =>  X_o = 2T - S
        ~ ifelse(.data$specialty_code == "Other", 2 * .x - sum(.x), .x)
      )
    ) |>
    dplyr::ungroup()
}

theatres_get_available <- function(qmco_data_path) {
  readxl::read_excel(qmco_data_path, skip = 15) |>
    dplyr::slice(-(1:3)) |>
    dplyr::select(
      org_code = 3,
      theatres = 5,
      day_case_theatres = 6
    )
}

theatres_generate_synthetic <- function(theatres_four_hour_sessions, theatres_available, path = "data") {
  four_hour_sessions <- theatres_four_hour_sessions |>
    dplyr::group_by(.data$specialty_code) |>
    dplyr::summarise(
      dplyr::across("four_hour_sessions", median),
      org_code = "synthetic"
    )

  available <- theatres_available |>
    dplyr::summarise(
      dplyr::across(where(is.numeric), purrr::compose(round, mean)),
      org_code = "synthetic"
    )

  theatres_save_data(four_hour_sessions, available, "synthetic", path)
}

theatres_save_data <- function(theatres_four_hour_sessions, theatres_available, params) {
  trust <- params$name

  four_hour_sessions <- theatres_four_hour_sessions |>
    dplyr::filter(.data$org_code %in% params$providers) |>
    dplyr::count(.data$specialty_code, wt = .data$four_hour_sessions) |>
    tibble::deframe() |>
    as.list()

  available <- theatres_available |>
    dplyr::filter(.data$org_code %in% params$providers) |>
    dplyr::select(-"org_code") |>
    tidyr::pivot_longer(tidyselect::everything()) |>
    tibble::deframe()

  fn <- file.path(params$path, trust, "theatres.json")

  c(available, list(four_hour_sessions = four_hour_sessions)) |>
    jsonlite::write_json(fn, pretty = TRUE, auto_unbox = TRUE)

  fn
}
