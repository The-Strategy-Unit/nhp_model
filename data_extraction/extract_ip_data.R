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

  inpatients <- tb_inpatients |>
    dplyr::arrange(.data$EPIKEY) |>
    dplyr::collect() |>
    janitor::clean_names() |>
    dplyr::mutate(
      rn = dplyr::row_number(),
      .before = tidyselect::everything()
    )

  strategy_lookups <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling_reference", "strategy_lookups"))

  strategies <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "strategies")) |>
    dplyr::semi_join(tb_inpatients, by = "EPIKEY") |>
    dplyr::inner_join(strategy_lookups, by = c("strategy")) |>
    dplyr::select("EPIKEY", "strategy", "strategy_type", "sample_rate") |>
    dplyr::collect() |>
    janitor::clean_names() |>
    dplyr::inner_join(
      dplyr::select(inpatients, "epikey", "rn"),
      by = "epikey"
    ) |>
    dplyr::relocate("rn", .before = tidyselect::everything()) |>
    dplyr::select(-"epikey")

  list(
    data = inpatients,
    strategies = strategies
  )
}

extract_ip_sample_data <- function(start_date, end_date, ...) {
  con <- DBI::dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  # only select organisations that had on average at least 50 people admitted
  # per day and also had day cases
  tbl_providers_of_interest <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling_reference", "org_code_type")) |>
    dplyr::filter(
      .data$org_type == "Acute",
      .data$org_subtype %in% c("Small", "Medium", "Large")
    ) |>
    dplyr::select("org_code")

  tbl_inpatients <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "inpatients")) |>
    dplyr::filter(
      .data$DISDATE >= start_date,
      .data$DISDATE <= end_date
    ) |>
    dplyr::semi_join(tbl_providers_of_interest, by = c("PROCODE3" = "org_code"))

  n_rows <- tbl_inpatients |>
    dplyr::count(.data$PROCODE3) |>
    dplyr::collect() |>
    dplyr::pull(.data$n) |>
    median() |>
    round()

  main_icb_rate <- tbl_inpatients |>
    dplyr::summarise(
      dplyr::across("is_main_icb", ~ mean(.x * 1.0, na.rm = TRUE))
    ) |>
    dplyr::collect() |>
    dplyr::pull("is_main_icb")

  inpatients <- tbl_inpatients |>
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
    dplyr::mutate(
      rn = dplyr::row_number(),
      .before = tidyselect::everything()
    )

  ip_sample <- dplyr::copy_to(
    con,
    dplyr::select(inpatients, "epikey"),
    "inpatients_random_sample"
  )

  strategy_lookups <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling_reference", "strategy_lookups"))

  strategies <- dplyr::tbl(con, dbplyr::in_schema("nhp_modelling", "strategies")) |>
    dplyr::semi_join(ip_sample, by = c("EPIKEY" = "epikey")) |>
    dplyr::inner_join(strategy_lookups, by = c("strategy")) |>
    dplyr::select("EPIKEY", "strategy", "strategy_type", "sample_rate") |>
    dplyr::collect() |>
    janitor::clean_names() |>
    dplyr::inner_join(
      dplyr::select(inpatients, "epikey", "rn"),
      by = "epikey"
    ) |>
    dplyr::relocate("rn", .before = tidyselect::everything()) |>
    dplyr::select(-"epikey")

  list(
    data = inpatients,
    strategies = strategies
  )
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
    dplyr::select(-"epikey", -"person_id", -"lsoa11", -"sushrg") |>
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

create_ip_synth_from_data <- function(data) {
  data |>
    dplyr::group_by(.data$classpat) |>
    dplyr::mutate(
      dplyr::across(c("age", "admidate"), ~ .x + sample(-5:5, dplyr::n(), TRUE)),
      dplyr::across(c("speldur", "imd04_decile", "ethnos"), ~ sample(.x, dplyr::n(), TRUE)),
      speldur = ifelse(.data$classpat == 5 & .data$speldur > 10, 10, .data$speldur),
      disdate = .data$admidate + .data$speldur,
      # "fix" age field
      age = dplyr::case_when(
        hsagrp == "birth" ~ 0L,
        age < 0L ~ 0L,
        age > 90L ~ 90L,
        TRUE ~ age
      )
    ) |>
    dplyr::ungroup() |>
    # randomly shuffle the specialty: ignore paeds/maternity/birth, and handle
    # each classpat separately
    (\(data) {
      a <- data |>
        dplyr::filter(.data$hsagrp %in% c("paeds", "maternity", "birth"))

      b <- data |>
        dplyr::anti_join(a, by = "rn") |>
        dplyr::group_by(.data$classpat) |>
        dplyr::mutate(
          dplyr::across(c("mainspef", "tretspef"), ~ sample(.x, dplyr::n(), TRUE))
        )

      dplyr::bind_rows(a, b)
    })()
}

union_bed_days_rows <- function(start_date) {
  force(start_date)

  function(data) {
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
}

save_ip_data <- function(data, strategies, name, path) {
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

  # for los reduction we have the general los reduction for all elective/non-elective rows inpatient
  general_los_reduction <- data |>
    dplyr::filter(.data$classpat == "1", .data$group %in% c("elective", "non-elective")) |>
    dplyr::transmute(
      .data$rn,
      strategy_type = "los reduction",
      strategy = paste0("general_los_reduction_", .data$group)
    )

  strategies_to_remove <- dplyr::bind_rows(
    # bads records where we wont convert daycases
    strategies |>
      dplyr::filter(
        .data$strategy |> stringr::str_detect("bads_"),
        .data$strategy != "bads_outpatients"
      ) |>
      dplyr::semi_join(
        data |>
          dplyr::filter(.data$classpat != 1),
        by = "rn"
      ),
    strategies |>
      dplyr::filter(strategy == "improved_discharge_planning_emergency")
  )

  strategies_fns <- strategies |>
    dplyr::anti_join(strategies_to_remove, by = c("rn", "strategy")) |>
    tidyr::drop_na("strategy_type") |>
    dplyr::bind_rows(general_los_reduction) |>
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
      fn <- file.path(path, name, glue::glue("ip_{strategy_type}_strategies.parquet"))

      data |>
        dplyr::arrange(.data$strategy, .data$rn) |>
        arrow::write_parquet(fn)

      fn
    })

  c(ip_fn, strategies_fns)
}

# ------------------------------------------------------------------------------

create_ip_extract <- function(start_date,
                              end_date,
                              providers,
                              specialties,
                              name,
                              extract_fn,
                              synth_fn,
                              path,
                              ...) {
  inpatients <- strategies <- NULL # for lint
  c(inpatients, strategies) %<-% extract_fn(start_date, end_date, providers)

  # partially apply the function
  ubdr <- union_bed_days_rows(start_date)

  inpatients |>
    create_ip_data(specialties) |>
    synth_fn() |>
    ubdr() |>
    save_ip_data(strategies, name, path)
}

create_synthetic_ip_extract <- function(start_date,
                                        end_date,
                                        ...,
                                        name = "synthetic",
                                        specialties = NULL,
                                        path = "data") {
  create_ip_extract(
    start_date,
    end_date,
    NULL,
    specialties,
    name,
    extract_ip_sample_data,
    create_ip_synth_from_data,
    path,
    ...
  )
}

create_provider_ip_extract <- function(params, specialties = NULL) {
  cat(paste("    [ip] running:", params$name))

  create_ip_extract(
    params$start_date,
    params$end_date,
    params$providers,
    specialties,
    params$name,
    extract_ip_data,
    identity,
    params$path,
  )
}
