#' @export
run_model <- function(params_json) {
  # load the params
  params <- jsonlite::read_json(params_json)

  strategy_params <- purrr::map_dfr(
    params$strategy_params, dplyr::bind_cols,
    .id = "strategy"
  ) |>
      dplyr::mutate(
        dplyr::across(
          tidyselect:::where(is.numeric),
          tidyr::replace_na,
          1
        )
      )

  demog_factors_fn <- do.call(demographic_factors, params$demographic_factors)

  data <- arrow::read_parquet(
    file.path(
      getOption("nhp_model_data_path", "test/data"),
      paste0(params$input_data, ".parquet")
    )
  )

  strategies <- arrow::read_parquet(
    file.path(
      getOption("nhp_model_data_path", "test/data"),
      paste0(params$input_data, "_strategies.parquet")
    )
  )

  # run the model
  plan(multisession, workers = getOption("nhp_model_ncpus", 1))

  path <- file.path(
    getOption("nhp_model_results_path", "test/results"),
    params$path
  )
  m <- model(path, data, strategies, demog_factors_fn, strategy_params)

  r <- future_walk(
    seq(params$run_start, params$run_end),
    .options = furrr_options(seed = 1734 + params$run_start),
    .progress = TRUE,
    m
  )

  plan(sequential)

  unlink(params_json)

  invisible(NULL)
}