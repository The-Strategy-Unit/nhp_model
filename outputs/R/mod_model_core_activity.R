#' model_core_activity UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_model_core_activity_ui <- function(id) {
  ns <- NS(id)
  tagList(
    shinycssloaders::withSpinner(
      gt::gt_output(ns("core_activity"))
    )
  )
}

#' model_core_activity Server Functions
#'
#' @noRd
mod_model_core_activity_server <- function(id, selected_model_run_id, data_cache) {
  moduleServer(id, function(input, output, session) {

    atpmo <- get_activity_type_pod_measure_options()

    summarised_data <- reactive({
      id <- selected_model_run()
      cosmos_get_model_core_activity(id) |>
        inner_join(atpmo, by = c("pod", "measure" = "measures"))
    }) |>
      shiny::bindCache(selected_model_run_id(), cache = data_cache)

    output$core_activity <- gt::render_gt({
      summarised_data() |>
        dplyr::select(
          .data$activity_type_name,
          .data$pod_name,
          .data$measure,
          .data$baseline,
          .data$median,
          .data$lwr_ci,
          .data$upr_ci
        ) |>
        gt::gt(groupname_col = c("activity_type_name", "pod_name")) |>
        gt::fmt_integer(c("baseline", "median", "lwr_ci", "upr_ci")) |>
        gt::cols_label(
          "measure" = "Measure",
          "baseline" = "Baseline",
          "median" = "Central Estimate",
          "lwr_ci" = "Lower",
          "upr_ci" = "Upper"
        ) |>
        gt::tab_spanner(
          "90% Confidence Interval",
          c("lwr_ci", "upr_ci")
        ) |>
        gt_theme()
    })
  })
}
