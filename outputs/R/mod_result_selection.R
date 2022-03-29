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
    shiny::selectInput(ns("model_name"), "Model Name", NULL),
    shiny::selectInput(ns("model_run"), "Model Run", NULL)
  )
}

#' result_selection Server Functions
#'
#' @noRd
mod_result_selection_server <- function(id) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    datasets_path <- get_golem_config("datasets_path")

    get_values_for_dropdowns <- function(path) {
      d1 <- dir(path, full.names = TRUE)
      d2 <- dir(path)
      purrr::set_names(d1, d2)
    }

    # on load, update the datasets dropdown
    observe({
      datasets <- get_values_for_dropdowns(datasets_path)
      shiny::updateSelectInput(session, "dataset", choices = datasets)
    })

    observeEvent(input$dataset, {
      ds <- req(input$dataset)
      model_names <- get_values_for_dropdowns(file.path(ds, "results"))
      shiny::updateSelectInput(session, "model_name", choices = model_names)
    })

    observeEvent(input$model_name, {
      mn <- req(input$model_name)
      model_runs <- get_values_for_dropdowns(mn)
      shiny::updateSelectInput(session, "model_run", choices = model_runs)
    })

    data <- reactive({
      mr <- req(input$model_run)
      arrow::read_parquet(file.path(mr, "model_results.parquet")) |>
        # remove any "__.__" columns, these are usually pandas indexes
        dplyr::select(-tidyselect::matches("^\\_{2}.*\\_{2}$")) |>
        dplyr::mutate(
          dplyr::across(
            .data$age_group,
            forcats::fct_relevel,
            "0-4",
            "5-14",
            "15-34",
            "35-49",
            "50-64",
            "65-84",
            "85+"
          )
        )
    })

    return(data)
  })
}

## To be copied in the UI
# mod_result_selection_ui("result_selection_ui_1")

## To be copied in the server
# mod_result_selection_server("result_selection_ui_1")