run_model <- function(name, ip, strategies, demog_factors, strategy_params, n = 100) {
  plan(multisession, workers = Sys.getenv("NCPUS"))

  m <- model(name, ip, strategies, strategy_params, demog_factors)

  r <- future_walk(
    1:n,
    .options = furrr_options(seed = 1734),
    .progress = TRUE,
    m
  )

  plan(sequential)

  invisible(NULL)
}