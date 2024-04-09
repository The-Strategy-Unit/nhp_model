.data <- NULL # lint helper
`%<-%` <- zeallot::`%<-%` # nolint

extract_ip_data <- function(start_date, end_date, providers) {
  con <- DBI::dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  tb_inpatients <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "inpatients")) |>
    dplyr::filter(
      .data$DISDATE >= start_date,
      .data$DISDATE <= end_date,
      .data$PROCODE3 %in% providers
    )

  tb_inpatients |>
    dplyr::arrange(.data$EPIKEY) |>
    dplyr::collect() |>
    janitor::clean_names() |>
    dplyr::rename(rn = "epikey")
}

# -------------------------------------------------------------------------------

create_ip_data <- function(inpatients, specialties) {
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

  fix_mainspef <- function(x) {
    dplyr::case_when(
      x == "199" ~ "100",
      x == "499" ~ "300",
      x == "560" ~ "501",
      x == "600" ~ "300",
      x == "601" ~ "141",
      x == "902" ~ "141",
      x == "903" ~ "141",
      x == "904" ~ "141",
      x == "950" ~ "300",
      x == "960" ~ "300",
      x == "NULL" ~ "300",
      x == "&" ~ "300",
      TRUE ~ x
    )
  }

  inpatients |>
    dplyr::arrange(.data$rn) |>
    dplyr::select(-"person_id", -"lsoa11", -"sushrg") |>
    dplyr::mutate(
      dplyr::across(tidyselect::ends_with("date"), lubridate::ymd),
      hsagrp = dplyr::case_when(
        .data$classpat %in% c("3", "4") ~ "reg",
        .data$admimeth %in% c("82", "83") ~ "birth",
        .data$mainspef == "420" ~ "paeds",
        (
          stringr::str_detect(.data$admimeth, "^3") | .data$mainspef %in% c("501", "560")
        ) & .data$age < 56L ~ "maternity",
        stringr::str_detect(.data$admimeth, "^2") ~ "emerg",
        .data$admimeth == "81" ~ "transfer",
        .data$admimeth %in% c("11", "12", "13") & .data$classpat == "1" ~ "ordelec",
        .data$admimeth %in% c("11", "12", "13") & .data$classpat == "2" ~ "daycase"
      ),
      dplyr::across("age", ~ pmin(.x, 90L)),
      tretspef_raw = .data[["tretspef"]],
      dplyr::across("tretspef", specialty_fn),
      dplyr::across("mainspef", fix_mainspef),
      dplyr::across(
        "imd04_decile",
        ~ forcats::fct_relevel(
          .x,
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
        )
      ),
      dplyr::across(c("age", "sex"), as.integer),
      is_wla = .data$admimeth == "11",
      group = dplyr::case_when(
        stringr::str_starts(.data$admimeth, "1") ~ "elective",
        stringr::str_starts(.data$admimeth, "3") ~ "maternity",
        TRUE ~ "non-elective"
      ),
      # handle case of maternity: make tretspef always Other (Medical)
      dplyr::across(
        "tretspef",
        ~ ifelse(.data$group == "maternity", "Other (Medical)", .x)
      )
    ) |>
    dplyr::select(-"procode3", -"fyear", -"icb22cdh") |>
    dplyr::filter(.data$sex %in% c(1, 2)) |>
    tidyr::drop_na("hsagrp", "speldur")
}

union_bed_days_rows <- function(data, start_date) {
  dplyr::bind_rows(
    .id = "bedday_rows",
    "FALSE" = data,
    "TRUE" = data |>
      dplyr::filter(.data$admidate < start_date) |>
      dplyr::mutate(
        dplyr::across(dplyr::ends_with("date"), ~ lubridate::`%m+%`(.x, lubridate::years(1)))
      )
  ) |>
    dplyr::mutate(dplyr::across("bedday_rows", as.logical)) |>
    dplyr::relocate("bedday_rows", .after = dplyr::everything())
}

save_ip_data <- function(data, name, path) {
  ip_fn <- file.path(path, name, "ip.parquet")

  data |>
    dplyr::arrange(
      .data$tretspef,
      .data$mainspef,
      .data$hsagrp,
      .data$sex,
      .data$ethnos,
      .data$age,
      .data$imd04_decile,
      .data$admidate
    ) |>
    arrow::write_parquet(ip_fn)

  ip_fn
}

# ------------------------------------------------------------------------------

create_provider_ip_extract <- function(params, specialties = NULL) {
  cat(paste("    [ip] running:", params$name))

  extract_ip_data(params$start_date, params$end_date, params$providers) |>
    create_ip_data(specialties) |>
    union_bed_days_rows(params$start_date) |>
    save_ip_data(params$name, params$path)
}

create_provider_ip_strategies <- function(params) {
  cat(paste("    [ip (strategies)] running:", params$name))

  con <- DBI::dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  strategy_lookups <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling_reference", "strategy_lookups")) |>
    dplyr::filter(!is.na(.data[["strategy_type"]]))

  tb_inpatients <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "inpatients")) |>
    dplyr::filter(
      .data$DISDATE >= !!(params$start_date),
      .data$DISDATE <= !!(params$end_date),
      .data$PROCODE3 %in% !!(params$providers)
    )

  tb_strategies <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "strategies")) |>
    dplyr::semi_join(tb_inpatients, by = "EPIKEY") |>
    dplyr::inner_join(strategy_lookups, by = c("strategy")) |>
    dplyr::select(rn = "EPIKEY", "strategy", "strategy_type", "sample_rate") |>
    # old strategy that should be removed
    dplyr::filter(.data$strategy != "improved_discharge_planning_emergency")

  # bads records where we wont convert daycases
  tb_bads_strategies_to_remove <- tb_strategies |>
    dplyr::filter(
      .data$strategy %LIKE% "bads_%", # nolint
      .data$strategy != "bads_outpatients"
    ) |>
    dplyr::semi_join(
      tb_inpatients |>
        dplyr::filter(.data$classpat != "1") |>
        dplyr::select("rn" = "EPIKEY"),
      by = "rn"
    )

  tb_strategies |>
    dplyr::anti_join(tb_bads_strategies_to_remove, by = c("rn", "strategy")) |>
    dplyr::collect() |>
    dplyr::group_nest(.data$strategy_type) |>
    dplyr::mutate(
      dplyr::across(
        "strategy_type",
        ~ forcats::fct_recode(
          .x,
          "activity_avoidance" = "admission avoidance",
          "efficiencies" = "los reduction"
        )
      )
    ) |>
    purrr::pmap_chr(\(strategy_type, data) {
      fn <- file.path(params$path, params$name, glue::glue("ip_{strategy_type}_strategies.parquet"))

      data |>
        dplyr::arrange(.data$strategy, .data$rn) |>
        arrow::write_parquet(fn)

      fn
    })
}
