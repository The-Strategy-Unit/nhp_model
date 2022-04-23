#' process_param_file
#'
#' @description A fct function
#'
#' @return The return value, if any, from executing the function.
#'
#' @noRd
process_param_file <- function(path, input_data, demographics_file, scenario_name) {

  data <- c(
    "run_settings",
    "dsi_wl",
    "pc_pg",
    "pc_hsa",
    "am_a_ip",
    "am_a_op",
    "am_a_aae",
    "am_tc_ip",
    "am_tc_op",
    "am_e_ip"
  ) |>
    set_names() |>
    map(read_excel, path = path)

  params <- list(
    name = scenario_name,
    input_data = input_data,
    seed = sample(1:1e5, 1),
    model_runs = data$run_settings$n_iterations,
    start_year = lubridate::year(data$run_settings$baseline_year),
    end_year = lubridate::year(data$run_settings$model_year),
    demographic_factors = list(
      file = demographics_file,
      variant_probabilities = data$pc_pg |>
        filter(probability > 0) |>
        deframe() |>
        as.list()
    ),
    health_status_adjustment = list(
      min_age = min(data$pc_hsa$age),
      max_age = max(data$pc_hsa$age),
      intervals = pmap(data$pc_hsa, \(lo, hi, ...) c(lo, hi))
    ),
    waiting_list_adjustment = list(
      inpatients = pmap(data$dsi_wl, \(lo, hi, ...) c(lo, hi)) |> set_names(data$dsi_wl$tretspef)
    )
  )

  params$strategy_params <- list(
    "admission_avoidance" = data$am_a_ip |>
      filter(include != 0) |>
      rowwise() |>
      transmute(strategy, interval = list(c(lo, hi))) |>
      deframe() |>
      map(~list(interval = .x))
  )

  params$strategy_params$los_reduction <- c(
    data$am_e_ip |>
      filter(include != 0) |>
      mutate(type = case_when(
        strategy |> str_starts("ambulatory_emergency_care") ~ "aec",
        strategy |> str_starts("pre-op") ~ "pre-op",
        TRUE ~ "all"
      )) |>
      rowwise() |>
      transmute(strategy, value = list(list(type = type, interval = list(c(lo, hi))))) |>
      deframe(),
    data$am_tc_ip |>
      filter(include != 0) |>
      mutate(target_type = ifelse(
        strategy |> str_detect("outpatients"),
        "outpatients",
        "daycase"
      )) |>
      rowwise() |>
      transmute(strategy, value = list(
        list(
          type = "bads",
          target_type = target_type,
          interval = list(c(lo, hi)),
          baseline_target_rate = baseline_target_rate,
          op_dc_split = op_dc_split
        )
      )) |>
      deframe()
  )

  params$outpatient_factors <- bind_rows(data[c("am_a_op", "am_tc_op")]) |>
    filter(include != 0) |>
    rowwise() |>
    transmute(strategy, value = list(list(interval = list(c(lo, hi))))) |>
    separate(strategy, c("strategy", "sub_group"), "\\|") |>
    group_nest(strategy) |>
    mutate(across(data, map, deframe)) |>
    deframe()

  params$aae_factors <- data$am_a_aae |>
    filter(include != 0) |>
    rowwise() |>
    transmute(strategy, value = list(list(interval = list(c(lo, hi))))) |>
    separate(strategy, c("strategy", "sub_group"), "\\|") |>
    group_nest(strategy) |>
    mutate(across(data, map, deframe)) |>
    deframe()


  params |> jsonlite::toJSON(auto_unbox = TRUE, pretty = TRUE)

  # should save this file...

}
