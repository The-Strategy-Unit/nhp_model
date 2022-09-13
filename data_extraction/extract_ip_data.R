library(tidyverse)
library(dbplyr)
library(DBI)
library(arrow)
library(janitor)
library(zeallot)

extract_ip_data <- function(providers) {
  con <- dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(DBI::dbDisconnect(con))

  tb_inpatients <- tbl(con, in_schema("nhp_modelling", "inpatients")) %>%
    filter(FYEAR == 201819, ADMIAGE <= 120, PROCODE3 %in% providers)

  inpatients <- tb_inpatients %>%
    arrange(EPIKEY) %>%
    collect() %>%
    clean_names() %>%
    mutate(rn = row_number(), .before = everything())

  strategy_lookups <- tbl(con, in_schema("nhp_modelling", "strategy_lookups"))

  strategies <- tbl(con, in_schema("nhp_modelling", "strategies")) %>%
    semi_join(tb_inpatients, by = "EPIKEY") %>%
    inner_join(strategy_lookups, by = c("strategy" = "full_strategy")) %>%
    select(EPIKEY, strategy = grouped_strategy, strategy_type, sample_rate) %>%
    collect() %>%
    clean_names() %>%
    inner_join(select(inpatients, epikey, rn), by = "epikey") %>%
    relocate(rn, .before = everything()) %>%
    select(-epikey)

  list(
    data = inpatients,
    strategies = strategies
  )
}

extract_ip_sample_data <- function() {
  con <- dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR"), timeout = 10
  )
  withr::defer(dbDisconnect(con))

  # only select organisations that had on average at least 50 people admitted
  # per day and also had day cases
  providers_of_interest <- tbl(con, sql("
  SELECT
    i.PROCODE3
  FROM
    nhp_modelling.inpatients i
  WHERE
    i.FYEAR = 201819
  AND
    i.ADMIAGE <= 120
  AND
    i.PROCODE3 LIKE 'R%'
  GROUP BY
    i.PROCODE3
  HAVING
    SUM(CASE CLASSPAT WHEN '1' THEN 1 ELSE 0 END) > 100 * 365
  AND
    SUM(CASE CLASSPAT WHEN '2' THEN 1 ELSE 0 END) > 0
  AND
  -- make sure the average age is over 18, should exclude Children's hospitals
    AVG(ADMIAGE) > 18
  "))
  tbl_providers_of_interest <- copy_to(
    con, providers_of_interest, "providers_of_interest"
  )

  # in case we need to view which providers are selected
  # ods <- tbl(con, sql("SELECT [Organisation_Code], [Organisation_Name]
  #                      FROM   [UK_Health_Dimensions].[ODS].[NHS_Trusts_SCD]
  #                      WHERE  [Is_Latest] = 1")) %>%
  #   collect()
  #
  # collect(tbl_providers_of_interest) %>%
  #   left_join(ods, by = c("PROCODE3" = "Organisation_Code")) %>%
  #   arrange(Organisation_Name) %>%
  #   View()

  tbl_inpatients <- tbl(con, in_schema("nhp_modelling", "inpatients")) %>%
    filter(FYEAR == 201819, ADMIAGE <= 120) %>%
    semi_join(tbl_providers_of_interest, by = "PROCODE3")

  n_rows <- tbl_inpatients %>%
    count(PROCODE3) %>%
    collect() %>%
    pull(n) %>%
    median() %>%
    round()

  # HAVE TO USE %>% rather than |>
  inpatients <- tbl_inpatients %>%
    arrange(x = NEWID()) %>%
    head(n_rows) %>%
    collect() %>%
    clean_names() %>%
    mutate(rn = row_number(), .before = everything())

  ip_sample <- copy_to(
    con, select(inpatients, epikey), "inpatients_random_sample"
  )

  strategy_lookups <- tbl(con, in_schema("nhp_modelling", "strategy_lookups"))

  strategies <- tbl(con, in_schema("nhp_modelling", "strategies")) %>%
    semi_join(ip_sample, by = c("EPIKEY" = "epikey")) %>%
    inner_join(strategy_lookups, by = c("strategy" = "full_strategy")) %>%
    select(EPIKEY, strategy = grouped_strategy, strategy_type, sample_rate) %>%
    collect() %>%
    clean_names() %>%
    inner_join(select(inpatients, epikey, rn), by = "epikey") %>%
    relocate(rn, .before = everything()) %>%
    select(-epikey)

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
      case_when(
        .x %in% specialties ~ .x,
        str_detect(.x, "^1(?!80|9[02])") ~
          "Other (Surgical)",
        str_detect(.x, "^(1(80|9[02])|[2346]|5(?!60)|83[134])") ~
          "Other (Medical)",
        TRUE ~ "Other"
      )
    }
  }

  fix_mainspef <- function(x) {
    case_when(
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
    arrange(rn) |>
    select(-epikey, -hesid, -sushrg) |>
    rename(age = admiage) |>
    mutate(
      across(ends_with("date"), lubridate::ymd),
      hsagrp = case_when(
        classpat %in% c("3", "4") ~ "reg",
        admimeth %in% c("82", "83") ~ "birth",
        mainspef == "420" ~ "paeds",
        (
          str_detect(admimeth, "^3") | mainspef %in% c("501", "560")
        ) & age < 56L ~ "maternity",
        str_detect(admimeth, "^2") ~ "emerg",
        admimeth == "81" ~ "transfer",
        admimeth %in% c("11", "12", "13") & classpat == "1" ~ "ordelec",
        admimeth %in% c("11", "12", "13") & classpat == "2" ~ "daycase"
      ),
      across(age, pmin, 90L),
      across(tretspef, specialty_fn),
      across(mainspef, fix_mainspef),
      across(
        imd04_decile,
        fct_relevel,
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
      across(c(age, sex), as.integer),
      admigroup = case_when(
        str_starts(admimeth, "1") ~ "elective",
        str_starts(admimeth, "3") ~ "maternity",
        TRUE ~ "non-elective"
      )
    ) |>
    filter(sex %in% c(1, 2)) |>
    drop_na(hsagrp, speldur)
}

create_ip_synth_from_data <- function(data) {
  data |>
    group_by(classpat) |>
    mutate(
      across(c(age, admidate), ~ .x + sample(-5:5, n(), TRUE)),
      across(c(speldur, imd04_decile, ethnos), ~ sample(.x, n(), TRUE)),
      speldur = ifelse(classpat == 5 & speldur > 10, 10, speldur),
      disdate = admidate + speldur,
      # "fix" age field
      age = case_when(
        hsagrp == "birth" ~ 0L,
        age < 0L ~ 0L,
        age > 90L ~ 90L,
        TRUE ~ age
      )
    ) |>
    ungroup() |>
    select(-procode3, -fyear) |>
    # randomly shuffle the specialty: ignore paeds/maternity/birth, and handle
    # each classpat separately
    (\(.data) {
      a <- .data |>
        filter(hsagrp %in% c("paeds", "maternity", "birth"))

      b <- .data |>
        anti_join(a, by = "rn") |>
        group_by(classpat) |>
        mutate(across(c(mainspef, tretspef), ~ sample(.x, n(), TRUE)))

      bind_rows(a, b)
    })()
}

save_ip_data <- function(data, strategies, name) {
  path <- function(...) file.path("data", name, ...)

  if (!dir.exists(path())) {
    dir.create(path())
  }

  data |>
    arrange(
      tretspef, mainspef, hsagrp, sex, ethnos, age, imd04_decile, admidate
    ) |>
    write_parquet(path("ip.parquet"))

  null_strats <- transmute(data, rn, strategy = "NULL", sample_rate = 1)

  strategies_to_remove <- bind_rows(
    # bads records where we wont convert daycases
    strategies |>
      filter(
        strategy |> str_detect("bads_"),
        strategy != "bads_outpatients"
      ) |>
      semi_join(data |> filter(classpat != 1), by = "rn"),
    strategies |>
      filter(strategy == "improved_discharge_planning_emergency")
  )

  strategies |>
    anti_join(strategies_to_remove, by = c("rn", "strategy")) |>
    drop_na(strategy_type) |>
    group_nest(strategy_type) |>
    mutate(across(strategy_type, str_replace, " ", "_")) |>
    pwalk(\(strategy_type, data) {
      t <- paste0(strategy_type, "_strategy")
      data |>
        bind_rows(null_strats) |>
        arrange(strategy, rn) |>
        rename({{ t }} := strategy) |>
        write_parquet(path(glue::glue("ip_{strategy_type}_strategies.parquet")))
    })

  cat(file.size(path("ip.parquet")) / 1024^2, "\n")

  invisible(list(data, strategies))
}

# ------------------------------------------------------------------------------

create_synthetic_ip_extract <- function(...,
                                        name = "synthetic",
                                        specialties = NULL) {
  c(inpatients, strategies) %<-% extract_ip_sample_data()

  inpatients |>
    create_ip_data(specialties) |>
    create_ip_synth_from_data() |>
    save_ip_data(strategies, "synthetic")
}

create_provider_ip_extract <- function(providers,
                                       ...,
                                       name = paste(providers, collapse = "_"),
                                       specialties = NULL) {
  cat(name, ":", sep = "")

  c(inpatients, strategies) %<-% extract_ip_data(providers)

  inpatients |>
    create_ip_data(specialties) |>
    save_ip_data(strategies, name)
}

# ------------------------------------------------------------------------------
rtt_specs <- c(
  "100", "101", "110", "120", "130", "140", "160", "300", "320", "330", "400",
  "410", "430", "502"
)

# create_synthetic_ip_extract(specialties = rtt_specs)

purrr::walk(
  list(
    "RXC",
    "RN5",
    "RYJ",
    "RGP",
    "RNQ",
    "RD8",
    "RBZ",
    "RX1",
    "RHW",
    "RA9",
    "RGR",
    c("RXN", "RTX"),
    c("RH5", "RBA")
  ),
  create_provider_ip_extract,
  specialties = rtt_specs
)
