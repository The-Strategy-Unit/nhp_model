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
mod_model_core_activity_server <- function(id, selected_model_run, data_cache) {
  moduleServer(id, function(input, output, session) {

    atpmo <- get_activity_type_pod_measure_options()

    summarised_data <- reactive({
      c(ds, sc, cd) %<-% selected_model_run()

      d <- cosmos_get_model_core_activity(ds, sc, cd) |>
        # if there are any model runs that had no data, add them back in with a value of 0
        tidyr::complete(
          tidyr::nesting(pod, measure),
          .data$model_run,
          fill = list(value = 0)
        )

      baseline <- d |>
        dplyr::filter(.data$model_run == -1) |>
        dplyr::select(.data$pod, .data$measure, baseline = .data$value)

      model <- d |>
        dplyr::filter(.data$model_run > 0) |>
        dplyr::group_by(.data$pod, .data$measure) |>
        dplyr::summarise(
          mean = mean(.data$value),
          lwr.ci = quantile(.data$value, 0.05),
          upr.ci = quantile(.data$value, 0.95),
          .groups = "drop"
        )

      inner_join(baseline, model, by = c("pod", "measure")) |>
        inner_join(atpmo, by = c("pod", "measure" = "measures"))
    }) |>
      shiny::bindCache(selected_model_run(), cache = data_cache)

    output$core_activity <- gt::render_gt({
      summarised_data() |>
        dplyr::select(
          .data$activity_type_name,
          .data$pod_name,
          .data$measure,
          .data$baseline,
          .data$mean,
          .data$lwr.ci,
          .data$upr.ci
        ) |>
        gt::gt(groupname_col = c("activity_type_name", "pod_name")) |>
        gt::fmt_integer(c("baseline", "mean", "lwr.ci", "upr.ci")) |>
        gt::cols_label(
          "measure" = "Measure",
          "baseline" = "Baseline",
          "mean" = "Central Estimate",
          "lwr.ci" = "Lower",
          "upr.ci" = "Upper"
        ) |>
        gt::tab_spanner(
          "90% Confidence Interval",
          c("lwr.ci", "upr.ci")
        ) |>
        gt_theme()
    })
  })
}
