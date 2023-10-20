.data <- NULL # lint helper
where <- NULL # lint helper
`%m+%` <- lubridate::`%m+%` # nolint
`%m-%` <- lubridate::`%m-%` # nolint

kh03_get_files <- function(url) {
  rvest::read_html(url) |>
    rvest::html_nodes("a") |>
    purrr::keep(\(.x)
    rvest::html_text(.x) |>
      stringr::str_detect("NHS organisations in England, Quarter.*XLS")) |>
    (
      \(.x) {
        x <- rvest::html_attr(.x, "href")
        t <- rvest::html_text(.x) |>
          stringr::str_replace("^.*Quarter (.), (.{7}).*$", "\\2 Q\\1")

        purrr::map2(x, t, \(f, p) list(file = f, period = p))
      })()
}

kh03_get_file <- function(x) {
  file <- x$file
  period <- x$period

  if (period >= "2013-14 Q4") {
    file_type <- "xlsx"
    skip_rows <- 14
  } else {
    file_type <- "xls"
    skip_rows <- if (period >= "2010-11 Q3") 13 else 3
  }

  kh03 <- withr::local_file(paste0("kh03.", file_type))
  download.file(file, kh03, quiet = TRUE, mode = "wb")

  overall <- readxl::read_excel(kh03, "NHS Trust by Sector", skip = 17, col_names = c(
    "year", "period_end", "skip_1", "org_code", "org_name", "skip_2",
    "available_general_and_acute", "available_learning_disabilities", "available_maternity", "available_mental_illness",
    "skip_3", "skip_4",
    "occupied_general_and_acute", "occupied_learning_disabilities", "occupied_maternity", "occupied_mental_illness",
    "skip_5", "skip_6", "skip_7", "skip_8", "skip_9", "skip_10"
  )) |>
    dplyr::select(-tidyselect::matches("skip_\\d+")) |>
    tidyr::pivot_longer(-("year":"org_name")) |>
    tidyr::separate("name", c("type", "specialty_group"), extra = "merge") |>
    tidyr::drop_na("value") |>
    tidyr::pivot_wider(names_from = "type", values_from = "value")

  by_specialty <- readxl::read_excel(kh03, "Occupied by Specialty", skip = skip_rows) |>
    dplyr::select(-1, -2, -3, -5) |>
    dplyr::rename(org_code = 1) |>
    tidyr::drop_na("org_code") |>
    tidyr::pivot_longer(-"org_code", names_to = "specialty", values_to = "occupied") |>
    tidyr::separate("specialty", c("specialty_code", "specialty_name"), extra = "merge")

  specialty_groups <- list(
    "maternity" = c("501"),
    "learning_disabilities" = c("700"),
    "mental_illness" = c("710", "711", "712", "713", "715")
  ) |>
    tibble::enframe("specialty_group", "specialty_code") |>
    tidyr::unnest("specialty_code") |>
    dplyr::right_join(
      dplyr::distinct(by_specialty, .data$specialty_code),
      by = "specialty_code"
    ) |>
    dplyr::mutate(
      dplyr::across("specialty_group", ~ tidyr::replace_na(.x, "general_and_acute"))
    ) |>
    dplyr::arrange(.data$specialty_code)

  overall |>
    dplyr::rename(available_total = "available", occupied_total = "occupied") |>
    dplyr::filter(.data$available_total > 0 | .data$occupied_total > 0) |>
    dplyr::inner_join(specialty_groups, by = "specialty_group", relationship = "many-to-many") |>
    dplyr::inner_join(by_specialty, by = c("org_code", "specialty_code")) |>
    dplyr::filter(.data$occupied > 0) |>
    dplyr::group_nest(
      dplyr::across("year":"occupied_total"),
      .key = "by_specialty"
    ) |>
    dplyr::mutate(
      period_start = as.Date(
        paste("1", .data$period_end, stringr::str_sub(.data$year, 1, 4)),
        "%d %B %Y"
      ) %m-% months(2),
      period_end = .data$period_start %m+% months(3) %m-% lubridate::days(1),
      quarter = paste0(year, " Q", lubridate::quarter(period_end, fiscal_start = 4)),
      year = NULL
    ) |>
    dplyr::relocate("quarter", "period_start", .before = "period_end")
}

kh03_combine <- function(kh03_overnight, kh03_dayonly) {
  kh03_overnight |>
    dplyr::distinct() |>
    dplyr::left_join(
      dplyr::distinct(kh03_dayonly),
      by = c("quarter", "org_code", "specialty_group")
    ) |>
    dplyr::mutate(
      dplyr::across("available_dayonly", ~ tidyr::replace_na(.x, 0)),
      dplyr::across("available_total", ~ .x + .data$available_dayonly)
    ) |>
    dplyr::select(-"available_dayonly")
}

# ------------------------------------------------------------------------------
# processing required for NHP work
# ------------------------------------------------------------------------------
kh03_process <- function(kh03_all, year) {
  fyear <- paste0(year, "-", year - 1999)
  kh03_all |>
    dplyr::filter(.data$quarter |> stringr::str_detect(fyear)) |>
    # use the overall occupancy rate to estimate available beds by specialty
    dplyr::mutate(rate = .data$occupied_total / .data$available_total) |>
    tidyr::unnest("by_specialty") |>
    dplyr::mutate(available = .data$occupied / .data$rate, .before = "occupied") |>
    # summarise rows to the year
    dplyr::group_by(
      dplyr::across("quarter", ~ stringr::str_to_lower(stringr::str_extract(.x, "Q\\d$"))),
      .data$org_code,
      .data$specialty_group,
      .data$specialty_code
    ) |>
    dplyr::summarise(
      dplyr::across(c("available", "occupied"), mean),
      .groups = "drop_last"
    ) |>
    # lump any specialty with < 1 occupied bed into the largest specialty
    dplyr::arrange(dplyr::desc(.data$occupied)) |>
    dplyr::mutate(
      dplyr::across("specialty_code", ~ ifelse(.data$occupied < 1, dplyr::first(.data$specialty_code), .x))
    ) |>
    # re-summarise the lumped data
    dplyr::group_by(.data$specialty_code, .add = TRUE) |>
    dplyr::summarise(
      dplyr::across(c("available", "occupied"), sum),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      year = year,
      .before = tidyselect::everything()
    )
}

# create a synthetic kh03 extract
kh03_generate_synthnetic <- function(kh03, path = "data") {
  data <- kh03 |>
    dplyr::filter(.data$specialty_group == "general_and_acute") |>
    dplyr::group_by(.data$org_code, .data$quarter) |>
    dplyr::summarise(
      dplyr::across(where(is.numeric), sum),
      .groups = "drop_last"
    ) |>
    dplyr::summarise(
      dplyr::across(where(is.numeric), min),
      .groups = "drop"
    ) |>
    dplyr::filter(.data$available |> dplyr::between(600, 900)) |>
    dplyr::semi_join(x = kh03, by = "org_code") |>
    tidyr::complete(
      .data$quarter,
      .data$org_code,
      tidyr::nesting(specialty_group, specialty_code),
      fill = list(available = 0, occupied = 0)
    ) |>
    dplyr::group_by(
      .data$quarter,
      dplyr::across(tidyselect::starts_with("specialty"))
    ) |>
    dplyr::summarise(
      dplyr::across(where(is.numeric), purrr::compose(round, mean)),
      .groups = "drop_last"
    ) |>
    dplyr::filter(.data$occupied >= 1)

  fn <- file.path(path, "synthetic", "kh03.csv")
  readr::write_csv(data, fn)

  fn
}

kh03_save_trust <- function(kh03, params) {
  trust <- params$name

  data <- kh03 |>
    dplyr::filter(
      .data[["org_code"]] %in% params$providers,
      .data[["year"]] == lubridate::year(params$start_date)
    ) |>
    dplyr::select(-"org_code", -"year") |>
    dplyr::group_by(
      dplyr::across(where(is.character))
    ) |>
    dplyr::summarise(
      dplyr::across(where(is.numeric), purrr::compose(round, sum)),
      .groups = "drop"
    )

  fn <- file.path(params$path, trust, "kh03.csv")
  readr::write_csv(data, fn)

  fn
}
