#' result_selection UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_result_selection_ui <- function(id) {
  ns <- shiny::NS(id)
  tagList(
    shiny::selectInput(ns("dataset"), "Dataset", NULL),
    shiny::selectInput(ns("scenario"), "Scenario", NULL),
    shiny::selectInput(ns("create_datetime"), "Model Run Time", NULL)
  )
}

#' result_selection Server Functions
#'
#' @noRd
mod_result_selection_server <- function(id) {
  moduleServer(id, function(input, output, session) {
    results_sets <- reactive({
      cosmos_get_result_sets() |>
        dplyr::relocate(.data$id, .after = tidyselect::everything()) |>
        dplyr::group_nest(.data$dataset, .data$scenario, .key = "create_datetime") |>
        dplyr::mutate(
          dplyr::across(.data$create_datetime, purrr::map, tibble::deframe)
        ) |>
        dplyr::group_nest(dataset, .key = "scenario") |>
        dplyr::mutate(
          dplyr::across(.data$scenario, purrr::map, tibble::deframe)
        ) |>
        tibble::deframe()
    })

    observe({
      datasets <- names(results_sets())
      shiny::updateSelectInput(session, "dataset", choices = datasets)
    })

    observe({
      ds <- shiny::req(input$dataset)
      scenarios <- names(results_sets()[[ds]])
      shiny::updateSelectInput(session, "scenario", choices = scenarios)
    })

    observe({
      ds <- shiny::req(input$dataset)
      sc <- shiny::req(input$scenario)

      labels <- \(.x) .x |>
        lubridate::as_datetime("%Y%m%d_%H%M%S", tz = "UTC") |>
        lubridate::with_tz() |>
        format("%d/%m/%Y %H:%M:%S")

      create_datetimes <- names(results_sets()[[ds]][[sc]]) |>
        purrr::set_names(labels)

      shiny::updateSelectInput(session, "create_datetime", choices = create_datetimes)
    })

    selected_model_run <- reactive({
      ds <- shiny::req(input$dataset)
      sc <- shiny::req(input$scenario)
      cd <- shiny::req(input$create_datetime)

      results_sets()[[ds]][[sc]][[cd]]
    })

    return(selected_model_run)
  })
}
