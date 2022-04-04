#' model_results_distribution UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_model_results_distribution_ui <- function(id) {
  ns <- NS(id)
  tagList(
    h1("Simulation Results"),
    mod_measure_selection_ui(ns("measure_selection"), FALSE),
    shinycssloaders::withSpinner(
      plotly::plotlyOutput(ns("simulation_results"))
    )
  )
}

#' model_results_distribution Server Functions
#'
#' @noRd
mod_model_results_distribution_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {
    filtered_data <- mod_measure_selection_server("measure_selection", data)

    selected_data <- reactive({
      d <- filtered_data()
      req(nrow(d) > 0)

      d |>
        dplyr::filter(.data$type != "principal") |>
        dplyr::group_by(.data$model_run) |>
        dplyr::summarise(dplyr::across(.data$value, sum), .groups = "drop")
    })

    output$simulation_results <- plotly::renderPlotly({
      d <- req(selected_data())
      req(nrow(d) > 0)

      b <- dplyr::filter(d, .data$model_run == -1)$value

      p <- d |>
        dplyr::filter(.data$model_run > 0) |>
        ggplot2::ggplot(aes(.data$value)) +
        # ggplot2::geom_histogram(fill = "#f9bf07", colour = "#2c2825", bins = 15) +
        ggplot2::geom_density(fill = "#f9bf07", colour = "#2c2825", alpha = 0.5) +
        ggplot2::geom_vline(xintercept = b)

      plotly::ggplotly(p)
    })
  })
}