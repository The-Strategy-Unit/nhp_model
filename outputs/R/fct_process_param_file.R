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
    purrr::set_names() |>
    map(readxl::read_excel, path = path)

  wla <- data$dsi_wl |>
    tibble::deframe() |>
    as.list()

  wla["X01"] <- wla["Other"]

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
    waiting_list_adjustment = list(inpatients = wla)
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
      dplyr::filter(include != 0) |>
      dplyr::mutate(type = case_when(
        strategy |> stringr::str_starts("ambulatory_emergency_care") ~ "aec",
        strategy |> stringr::str_starts("pre-op") ~ "pre-op",
        TRUE ~ "all"
      )) |>
      dplyr::rowwise() |>
      dplyr::transmute(strategy, value = list(list(type = type, interval = list(lo, hi)))) |>
      tibble::deframe(),
    data$am_tc_ip |>
      dplyr::filter(include != 0) |>
      dplyr::mutate(target_type = ifelse(
        strategy |> stringr::str_detect("outpatients"),
        "outpatients",
        "daycase"
      )) |>
      dplyr::rowwise() |>
      dplyr::transmute(strategy, value = list(
        list(
          type = "bads",
          target_type = target_type,
          interval = list(lo, hi),
          baseline_target_rate = baseline_target_rate,
          op_dc_split = op_dc_split
        )
      )) |>
      tibble::deframe()
  )

  params$outpatient_factors <- dplyr::bind_rows(data[c("am_a_op", "am_tc_op")]) |>
    dplyr::filter(include != 0) |>
    dplyr::rowwise() |>
    dplyr::transmute(strategy, value = list(list(interval = list(lo, hi)))) |>
    tidyr::separate(strategy, c("strategy", "sub_group"), "\\|") |>
    dplyr::group_nest(strategy) |>
    dplyr::mutate(dplyr::across(data, map, deframe)) |>
    tibble::deframe()

  params$aae_factors <- data$am_a_aae |>
    dplyr::filter(include != 0) |>
    dplyr::rowwise() |>
    dplyr::transmute(strategy, value = list(list(interval = list(lo, hi)))) |>
    tidyr::separate(strategy, c("strategy", "sub_group"), "\\|") |>
    dplyr::group_nest(strategy) |>
    dplyr::mutate(dplyr::across(data, purrr::map, tibble::deframe)) |>
    tibble::deframe()


  params
}
