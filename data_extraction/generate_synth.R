.data <- rlang::.data

create_ip_synth <- function(ip_files, specialties) {
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
        ),
        original_provider = "SYNTH",
        sitetret = sample(paste0("SYNTH0", 1:3), dplyr::n(), TRUE),
        resladst_ons = NULL
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
            dplyr::across(c("mainspef", "tretspef_raw"), ~ sample(.x, dplyr::n(), TRUE)),
            tretspef = specialty_fn(.data[["tretspef_raw"]])
          )

        dplyr::bind_rows(a, b)
      })()
  }

  paths <- dirname(ip_files)
  ip_s_a_files <- file.path(paths, "ip_activity_avoidance_strategies.parquet")
  ip_s_e_files <- file.path(paths, "ip_efficiencies_strategies.parquet")

  f <- file.exists(ip_files) & file.exists(ip_s_a_files) & file.exists(ip_s_e_files)
  ip_files <- ip_files[f]
  ip_s_a_files <- ip_s_a_files[f]
  ip_s_e_files <- ip_s_e_files[f]

  con <- DBI::dbConnect(duckdb::duckdb())
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE))

  ds <- arrow::open_dataset(ip_files) |>
    arrow::to_duckdb(table_name = "ip", con = con) |>
    dplyr::slice_sample(n = 1e5)

  ip <- ds |>
    dplyr::collect() |>
    create_ip_synth_from_data()

  ds_sample <- dplyr::copy_to(con, ip, "ip_sample")

  ds_s_a <- arrow::open_dataset(ip_s_a_files) |>
    arrow::to_duckdb(table_name = "ip_s_a", con = con) |>
    dplyr::semi_join(ds_sample, by = "rn") |>
    dplyr::summarise(
      .by = c("rn", "strategy"),
      dplyr::across("sample_rate", \(.x) min(.x, na.rm = TRUE))
    )

  ip_s_a <- dplyr::collect(ds_s_a)

  ds_s_e <- arrow::open_dataset(ip_s_e_files) |>
    arrow::to_duckdb(table_name = "ip_s_e", con = con) |>
    dplyr::semi_join(ds_sample, by = "rn") |>
    dplyr::summarise(
      .by = c("rn", "strategy"),
      dplyr::across("sample_rate", \(.x) min(.x, na.rm = TRUE))
    )

  ip_s_e <- dplyr::collect(ds_s_e)

  ip_path <- "data/2019/synthetic/ip.parquet"
  ip_s_a_path <- "data/2019/synthetic/ip_activity_avoidance_strategies.parquet"
  ip_s_e_path <- "data/2019/synthetic/ip_efficiencies_strategies.parquet"

  arrow::write_parquet(ip, ip_path)
  arrow::write_parquet(ip_s_a, ip_s_a_path)
  arrow::write_parquet(ip_s_e, ip_s_e_path)

  c(ip_path, ip_s_a_path, ip_s_e_path)
}

create_synth_gams <- function(ip_data, op_data, aae_data, demographic_factors) {
  create_gams <- function(base_year) {
    withr::local_envvar("RETICULATE_PYTHON" = "")

    reticulate::use_condaenv("nhp", conda = r"{C:\ProgramData\Miniconda3\Scripts\conda.exe}")
    hsa <- reticulate::import("model.hsa_gams")

    hsa$run("synthetic", base_year) |> # returns filename
      stringr::str_replace_all("\\\\", "/") # returns files with \, convert to /
  }

  callr::r(
    \(fn, b, ...) fn(b),
    args = list(
      fn = create_gams,
      b = "2019"
    )
  )
}
