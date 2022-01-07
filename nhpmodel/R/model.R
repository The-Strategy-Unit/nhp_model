#' Run Model
#'
#' @param model_run_name what to name the files
#' @param data the data to run the model on, should contain four columns (rn, admiage, sex, speldur)
#' @param strategies a table containing two columns (rn, strategy). Multiple strategies may be defined per rn
#' @param strategy_params a table containing five columns:
#'     (strategy, avoidance_low, avoidance_high, speldur_adjustment, strategy_weight)
#' @param demog_factors_fn a function which returns a table containing two columns (demogroup, factor)
#'
#' @return a table containing two columns (rn, speldur)
model <- function(model_run_name, run_number, data, strategies, strategy_params, demog_factors_fn) {
  # for each run select a single demographic factor set to use
  demog_factors <- demog_factors_fn()

  # for each row we pick a single strategy
  selected_strategy <- strategies |>
    group_by(rn) |>
    summarise(strategy = purrr::map2_chr(
      list(strategy),
      list(strategy_weight),
      # add in the NULL strategy as a selection possibility
      \(x, y) sample(c("null", x), 1, prob = c(1, y))
    )) |>
    filter(strategy != "null") |>
    arrange(rn)

  # pick an avoidance value for this model run
  strategy_avoidance_run <- strategy_params |>
    mutate(mean = (avoidance_high + avoidance_low) / 2,
            sd = (avoidance_high - avoidance_low) / (2 * qnorm(0.95))) |>
    transmute(strategy, factor = purrr::map2_dbl(mean, sd, rnorm, n = 1))

  strategy_speldur_run <- strategy_params |>
    select(strategy, factor = los_change)

  # helper function that takes the data, a set of parameters, and then the columns to group by / join on
  f <- function(x, params, ...) {
    cols <- purrr::map_chr(rlang::enquos(...), rlang::as_name)

    x |>
      group_by(...) |>
      summarise(across(rn, list), .groups = "drop") |>
      inner_join(params, by = cols) |>
      mutate(across(factor, `*`, purrr::map_dbl(rn, length)),
              across(rn, purrr::map2, factor, sample, replace = TRUE)) |>
      select(rn) |>
      tidyr::unnest(rn)
  }

  # select just the row number and speldur column
  data_speldur <- select(data, rn, speldur)

  results <- data |>
    # run the demographic factors part of the model
    f(demog_factors, sex, admiage) |>
    # run the avoidance strategies part of the model
    inner_join(selected_strategy, by = "rn") |>
    f(strategy_avoidance_run, strategy) |>
    # now calculate adjusted los
    inner_join(data_speldur, by = "rn") |>
    # we need to re-bring in the selected strategy
    inner_join(selected_strategy, by = "rn") |>
    inner_join(strategy_speldur_run, by = "strategy") |>
    mutate(across(speldur, `*`, factor)) |>
    # return just the relevant data
    select(rn, speldur)

  results_file <- glue::glue("/mnt/results/{model_run_name}/results/{run_number}.parquet")
  arrow::write_parquet(results, results_file)

  selected_strategy_file <- glue::glue("/mnt/results/{model_run_name}/selected_strategy/{run_number}.parquet")
  arrow::write_parquet(selected_strategy, selected_strategy_file)

  selected_variant_file <- glue::glue("/mnt/results/{model_run_name}/selected_variant/{run_number}.txt")
  writeLines(
    c("selected_variant", attr(demog_factors, "selected_variant")),
    selected_variant_file
  )

  invisible(NULL)
}
