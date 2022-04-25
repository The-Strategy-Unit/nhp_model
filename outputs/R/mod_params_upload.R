#' params_upload UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_params_upload_ui <- function(id) {
  ns <- shiny::NS(id)
  tagList(
    shiny::fileInput(ns("params_upload"), "Upload Params Excel File", accept = ".xlsx"),
    shiny::selectInput(ns("dataset"), "Dataset", choices = NULL),
    shiny::selectInput(
      ns("demographics_file"),
      "Demographics File",
      choices = c("default" = "demographic_factors.csv")
    ),
    shiny::textInput(ns("scenario_name"), "Scenario Name"),
    shiny::actionButton(ns("submit_run"), "Submit"),
    shiny::textOutput(ns("status"))
  )
}

#' params_upload Server Functions
#'
#' @noRd
mod_params_upload_server <- function(id) {
  moduleServer(id, function(input, output, session) {
    shiny::observe({
      shiny::updateSelectInput(session, "dataset", choices = c("synthetic", "RL4"))
    })

    status <- shiny::reactive("waiting")

    observeEvent(input$submit_run, {
      file <- shiny::req(input$params_upload)
      ext <- tools::file_ext(file$datapath)
      shiny::validate(shiny::need(ext == "xlsx", "Please upload an xlsx file"))

      dataset <- shiny::req(input$dataset)
      demographics_file <- shiny::req(input$demographics_file)
      scenario_name <- shiny::req(input$scenario_name)

      params_json <- process_param_file(file$datapath, dataset, demographics_file, scenario_name) |>
        jsonlite::toJSON(auto_unbox = TRUE, pretty = TRUE)

      status("submitting model run to batch")
      submit_model_run(params_json)
      status("submitted to batch")
    })

    output$status <- shiny::renderText(status())
  })
}