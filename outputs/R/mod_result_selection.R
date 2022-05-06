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
  con_str <- glue::glue(
    .sep = ";",
    "Driver={{ODBC Driver 18 for SQL Server}}",
    "server={Sys.getenv('DB_SERVER')}",
    "database={Sys.getenv('DB_DATABASE')}",
    "Authentication={Sys.getenv('DB_AUTHENTICATION_TYPE')}"
  )
  db_con <- DBI::dbConnect(odbc::odbc(), .connection_string = con_str)

  moduleServer(id, function(input, output, session) {
    dropdown_options <- reactive({
      dplyr::tbl(db_con, "aggregated_results") |>
        dplyr::distinct(.data$dataset, .data$scenario, .data$create_datetime) |>
        dplyr::arrange(.data$dataset, .data$scenario, .data$create_datetime) |>
        dplyr::collect()
    })

    observe({
      datasets <- dropdown_options() |>
        dplyr::pull(.data$dataset) |>
        unique()

      shiny::updateSelectInput(session, "dataset", choices = datasets)
    })

    observe(
      {
        ds <- req(input$dataset)

        scenarios <- dropdown_options() |>
          dplyr::filter(.data$dataset == ds) |>
          dplyr::pull(.data$scenario) |>
          unique()

        shiny::updateSelectInput(session, "scenario", choices = scenarios)
      },
      priority = 90
    )

    observe(
      {
        ds <- req(input$dataset)
        sc <- req(input$scenario)

        labels <- \(.x) .x |>
          lubridate::as_datetime("%Y%m%d_%H%M%S", tz = "UTC") |>
          format("%d/%m/%Y %H:%M:%S")

        create_datetimes <- dropdown_options() |>
          dplyr::filter(.data$dataset == ds, .data$scenario == sc) |>
          dplyr::pull(.data$create_datetime) |>
          unique() |>
          purrr::set_names(labels)

        shiny::updateSelectInput(session, "create_datetime", choices = create_datetimes)
      },
      priority = 80
    )

    selected_model_run <- reactive({
      ds <- req(input$dataset)
      sc <- req(input$scenario)
      cd <- req(input$create_datetime)

      valid_options_selected <- dropdown_options() |>
        dplyr::filter(.data$dataset == ds, .data$scenario == sc, .data$create_datetime == cd) |>
        nrow() == 1

      req(valid_options_selected)

      cat("loading data...")
      dfs <- list(
        data = get_data(db_con, ds, sc, cd),
        change_factors = get_change_factors(db_con, ds, sc, cd)
      )
      cat(" done\n\n")

      dfs
    })

    return(selected_model_run)
  })
}