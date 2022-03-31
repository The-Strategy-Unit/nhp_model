#' measure_selection UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_measure_selection_ui <- function(id, aggregation = TRUE) {
  ns <- NS(id)

  width <- if (aggregation) 3 else 4

  tagList(
    fluidRow(
      column(width, selectInput(ns("activity_type"), "Activity Type", NULL)),
      column(width, selectInput(ns("pod"), "POD", NULL)),
      column(width, selectInput(ns("measure"), "Measure", NULL)),
      # no matter what, we need to include this input, but we can hide it with CSS
      column(
        width,
        style = glue::glue("display: {if (aggregation) 'block' else 'none'}"),
        selectInput(
          ns("aggregation"),
          "Aggregation",
          c("Age Group", "Treatment Specialty")
        )
      )
    )
  )
}

#' measure_selection Server Functions
#'
#' @noRd
mod_measure_selection_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {
    # handle onload
    observe({
      d <- data()
      req(nrow(d) > 0)

      activity_types <- dataset_display |>
        dplyr::semi_join(data(), by = "dataset") |>
        (function(.x) purrr::set_names(.x[[1]], .x[[2]]))()

      shiny::updateSelectInput(session, "activity_type", choices = activity_types)
    })

    shiny::observeEvent(input$activity_type, {
      at <- req(input$activity_type)

      d <- data() |>
        dplyr::filter(.data$dataset == at)

      pods <- pod_display |>
        dplyr::semi_join(d, by = "pod") |>
        (function(.x) purrr::set_names(.x[[1]], .x[[2]]))()

      shiny::updateSelectInput(session, "pod", choices = pods)
    })

    shiny::observeEvent(input$pod, {
      at <- req(input$activity_type)
      p <- req(input$pod)
      d <- data() |>
        dplyr::filter(.data$dataset == at, .data$pod == p)

      measures <- measure_display |>
        dplyr::semi_join(d, by = "measure") |>
        (function(.x) purrr::set_names(.x[[1]], .x[[2]]))()

      shiny::updateSelectInput(session, "measure", choices = measures)
    })

    filtered_data <- reactive({
      at <- req(input$activity_type)
      p <- req(input$pod)
      m <- req(input$measure)

      a <- switch(req(input$aggregation),
        "Age Group" = "age_group",
        "Treatment Specialty" = "tretspef"
      )

      d <- data() |>
        dplyr::filter(.data$dataset == at, .data$pod == p, .data$measure == m) |>
        dplyr::group_by(.data$sex, agg = .data[[a]], .data$type, .data$model_run) |>
        dplyr::summarise(dplyr::across(.data$value, sum), .groups = "drop")

      attr(d, "aggregation") <- input$aggregation

      return(d)
    })

    return(filtered_data)
  })
}
