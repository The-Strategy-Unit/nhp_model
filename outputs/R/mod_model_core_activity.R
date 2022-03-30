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
mod_model_core_activity_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    summarised_data <- reactive({
      inner_join(
        data() |>
          dplyr::filter(.data$type == "baseline") |>
          dplyr::select(-.data$type) |>
          dplyr::group_by(
            .data$dataset,
            .data$pod,
            .data$measure
          ) |>
          dplyr::summarise(baseline = sum(.data$value), .groups = "drop"),
        data() |>
          dplyr::filter(.data$type == "model") |>
          dplyr::group_by(
            .data$dataset,
            .data$pod,
            .data$measure,
            .data$model_run
          ) |>
          dplyr::summarise(dplyr::across(.data$value, sum), .groups = "drop_last") |>
          # if there are any model runs that had no data, add them back in with a value of 0
          grouped_complete(
            tidyr::nesting(dataset, pod, measure),
            .data$model_run,
            fill = list(value = 0)
          ) |>
          dplyr::summarise(dplyr::across(.data$value, list), .groups = "drop") |>
          dplyr::mutate(dplyr::across(.data$value, purrr::map, DescTools::MeanCI)) |>
          tidyr::unnest_wider(.data$value),
        by = c("dataset", "pod", "measure")
      )
    })

    output$core_activity <- gt::render_gt({
      summarised_data() |>
        dplyr::inner_join(dataset_display, by = "dataset") |>
        dplyr::select(-.data$dataset) |>
        dplyr::inner_join(pod_display, by = "pod") |>
        dplyr::select(-.data$pod) |>
        dplyr::inner_join(measure_display, by = "measure") |>
        dplyr::select(-.data$measure) |>
        dplyr::relocate(tidyselect::ends_with("display"), .before = everything()) |>
        gt::gt(groupname_col = c("dataset_display", "pod_display")) |>
        gt::fmt_integer(c("baseline", "mean", "lwr.ci", "upr.ci")) |>
        gt::cols_label(
          "measure_display" = "Measure",
          "baseline" = "Baseline",
          "mean" = "Central Estimate",
          "lwr.ci" = "Lower",
          "upr.ci" = "Upper"
        ) |>
        gt::tab_spanner(
          "95% Confidence Interval",
          c("lwr.ci", "upr.ci")
        ) |>
        gt_theme()
    })
  })
}