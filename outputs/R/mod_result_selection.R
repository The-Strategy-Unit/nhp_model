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

    observe({
      datasets <- cosmos_get_datasets()

      shiny::updateSelectInput(session, "dataset", choices = datasets)
    })

    observe(
      {
        ds <- shiny::req(input$dataset)

        scenarios <- cosmos_get_scenarios(ds)

        shiny::updateSelectInput(session, "scenario", choices = scenarios)
      },
      priority = 90
    )

    observe(
      {
        ds <- shiny::req(input$dataset)
        sc <- shiny::req(input$scenario)

        labels <- \(.x) .x |>
          lubridate::as_datetime("%Y%m%d_%H%M%S", tz = "UTC") |>
          lubridate::with_tz() |>
          format("%d/%m/%Y %H:%M:%S")

        create_datetimes <- cosmos_get_create_datetimes(ds, sc) |>
          purrr::set_names(labels)

        shiny::updateSelectInput(session, "create_datetime", choices = create_datetimes)
      },
      priority = 80
    )

    selected_model_run <- reactive({
      ds <- shiny::req(input$dataset)
      sc <- shiny::req(input$scenario)
      cd <- shiny::req(input$create_datetime)

      list(dataset = ds, scenario = sc, create_datetime = cd)
    })

    return(selected_model_run)
  })
}
