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
    shiny::checkboxInput(ns("show_origin"), "Show Origin (zero)?"),
    shinycssloaders::withSpinner(
      plotly::plotlyOutput(ns("distribution"), height = "800px")
    )
  )
}

#' model_results_distribution Server Functions
#'
#' @noRd
mod_model_results_distribution_server <- function(id, selected_model_run, data_cache) {
  moduleServer(id, function(input, output, session) {
    filtered_data <- mod_measure_selection_server("measure_selection", selected_model_run, data_cache)

    selected_data <- reactive({
      d <- filtered_data()
      req(nrow(d) > 0)

      d |>
        dplyr::filter(.data$type != "principal") |>
        dplyr::group_by(.data$model_run, .data$variant) |>
        dplyr::summarise(dplyr::across(.data$value, sum), .groups = "drop")
    })

    output$distribution <- plotly::renderPlotly({
      d <- req(selected_data())
      req(nrow(d) > 0)

      b <- dplyr::filter(d, .data$model_run == -1)$value

      colour_scale <- ggplot2::scale_fill_manual(values = c(
        "TODO" = "#f9bf07",
        "principal" = "#f9bf07",
        "high_migration" = "#5881c1"
      ))


      p1 <- plotly::ggplotly({
        d |>
          dplyr::filter(.data$model_run > 0) |>
          dplyr::mutate(dplyr::across(.data$value, as.integer)) |>
          ggplot2::ggplot(aes(.data$value)) +
          ggplot2::geom_density(fill = "#f9bf07", colour = "#2c2825", alpha = 0.5) +
          ggplot2::geom_vline(xintercept = b) +
          ggplot2::expand_limits(x = ifelse(input$show_origin, 0, b)) +
          ggplot2::theme(
            axis.text = element_blank(),
            axis.title = element_blank(),
            axis.ticks = element_blank()
          )
      })

      p2 <- plotly::ggplotly({
        d |>
          dplyr::filter(.data$model_run > 0) |>
          dplyr::mutate(dplyr::across(.data$value, as.integer)) |>
          ggplot2::ggplot(aes("1", .data$value, colour = .data$variant)) +
          # ggplot2::geom_violin(show.legend = FALSE) +
          ggbeeswarm::geom_quasirandom(groupOnX = TRUE, alpha = 0.5) +
          ggplot2::geom_hline(yintercept = b) +
          ggplot2::expand_limits(y = ifelse(input$show_origin, 0, b)) +
          colour_scale +
          # have to use coord flip with boxplots/violin plots and plotly...
          ggplot2::coord_flip() +
          ggplot2::scale_y_continuous(labels = scales::comma) +
          ggplot2::theme(
            axis.text.y = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank()
          )
      })

      plotly::subplot(p1, p2, nrows = 2) |>
        plotly::layout(legend = list(
          orientation = "h"
        ))
    })
  })
}
