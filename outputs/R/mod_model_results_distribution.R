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
    fluidRow(
      column(4, selectInput(ns("activity_type"), "Activity Type", NULL)),
      column(4, selectInput(ns("pod"), "POD", NULL)),
      column(4, selectInput(ns("measure"), "Measure", NULL))
    ),
    plotly::plotlyOutput(ns("simulation_results"))
  )
}

#' model_results_distribution Server Functions
#'
#' @noRd
mod_model_results_distribution_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    shiny::observe({
      activity_types <- dataset_display |>
        dplyr::semi_join(data(), by = "dataset") |>
        (\(.x) purrr::set_names(.x[[1]], .x[[2]]))()

      shiny::updateSelectInput(session, "activity_type", choices = activity_types)
    })

    shiny::observeEvent(input$activity_type, {
      at <- req(input$activity_type)
      d <- data() |>
        dplyr::filter(.data$dataset == at)

      pods <- pod_display |>
        dplyr::semi_join(d, by = "pod") |>
        (\(.x) purrr::set_names(.x[[1]], .x[[2]]))()

      shiny::updateSelectInput(session, "pod", choices = pods)
    })

    shiny::observeEvent(input$pod, {
      at <- req(input$activity_type)
      p <- req(input$pod)
      d <- data() |>
        dplyr::filter(.data$dataset == at, .data$pod == p)

      measures <- measure_display |>
        dplyr::semi_join(d, by = "measure") |>
        (\(.x) purrr::set_names(.x[[1]], .x[[2]]))()

      shiny::updateSelectInput(session, "measure", choices = measures)
    })

    filtered_data <- reactive({
      at <- req(input$activity_type)
      p <- req(input$pod)
      m <- req(input$measure)

      data() |>
        dplyr::filter(.data$type != "principal", .data$dataset == at, .data$pod == p, .data$measure == m) |>
        dplyr::group_by(.data$model_run) |>
        dplyr::summarise(dplyr::across(.data$value, sum), .groups = "drop")
    })

    output$simulation_results <- plotly::renderPlotly({
      d <- req(filtered_data())
      req(nrow(d) > 0)

      b <- dplyr::filter(d, .data$model_run == 0)$value

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