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
  ns <- NS(id)
  tagList(
    shiny::selectInput(ns("dataset"), "Dataset", NULL),
    shiny::selectInput(ns("scenario"), "Scenario", NULL),
    shiny::selectInput(ns("model_run"), "Model Run Time", NULL)
  )
}

#' result_selection Server Functions
#'
#' @noRd
mod_result_selection_server <- function(id) {
  con_str <- glue::glue(
    .sep = ";",
    "Driver={{ODBC Driver 18 for SQL Server}}",
    "server={Sys.getenv('DB_SERVER')}",
    "database={Sys.getenv('DB_DATABASE')}",
    "Authentication={Sys.getenv('DB_AUTHENTICATION_TYPE')}"
  )
  db_con <- DBI::dbConnect(odbc::odbc(), .connection_string = con_str)

  moduleServer(id, function(input, output, session) {
     # on load, update the datasets dropdown
    observe({
      datasets <- dplyr::tbl(db_con, "aggregated_results") |>
        dplyr::distinct(.data$dataset) |>
        dplyr::pull(.data$dataset)

      shiny::updateSelectInput(session, "dataset", choices = datasets)
    })

    observeEvent(input$dataset, {
      ds <- req(input$dataset)

      scenarios <- dplyr::tbl(db_con, "aggregated_results") |>
        dplyr::filter(.data$dataset == ds) |>
        dplyr::distinct(.data$scenario) |>
        dplyr::pull(.data$scenario)

      shiny::updateSelectInput(session, "scenario", choices = scenarios)
    })

    observeEvent(input$scenario, {
      ds <- req(input$dataset)
      sc <- req(input$scenario)

      model_runs <- dplyr::tbl(db_con, "aggregated_results") |>
        dplyr::filter(.data$dataset == ds, .data$scenario == sc) |>
        dplyr::distinct(.data$create_datetime) |>
        dplyr::pull(.data$create_datetime)

      shiny::updateSelectInput(session, "model_run", choices = model_runs)
    })

    selected_model_run <- reactive({
      ds <- req(input$dataset)
      sc <- req(input$scenario)
      mr <- req(input$model_run)

      cat("loading data...")
      dfs <- list(
        data = get_data(db_con, ds, sc, mr),
        change_factors = get_change_factors(db_con, ds, sc, mr)
      )
      cat(" done\n\n")

      dfs
    })

    return(selected_model_run)
  })
}
