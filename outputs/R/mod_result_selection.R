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

  # create a 200 MiB cache on disk
  data_cache <- cachem::cache_disk(dir = ".cache/data_cache", max_size = 200 * 1024^2)

  moduleServer(id, function(input, output, session) {
    dropdown_options <- reactive({
      dplyr::tbl(db_con, "model_runs") |>
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
        ds <- shiny::req(input$dataset)

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
        ds <- shiny::req(input$dataset)
        sc <- shiny::req(input$scenario)

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
      ds <- shiny::req(input$dataset)
      sc <- shiny::req(input$scenario)
      cd <- shiny::req(input$create_datetime)

      options_selected <- dropdown_options() |>
        dplyr::filter(.data$dataset == ds, .data$scenario == sc, .data$create_datetime == cd)

      shiny::req(nrow(options_selected) == 1)

      cat("loading data...")
      dfs <- list(
        data = get_data(db_con, ds, sc, cd),
        change_factors = get_change_factors(db_con, ds, sc, cd),
        years = list(
          start_year = options_selected$start_year,
          end_year = options_selected$end_year
        )
      )
      cat(" done\n\n")

      dfs
    }) |>
      shiny::bindCache(input$dataset, input$scenario, input$create_datetime, cache = data_cache)

    return(selected_model_run)
  })
}