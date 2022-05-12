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
    shiny::selectInput(ns("dataset"), "Dataset", choices = NULL),
    shiny::selectInput(
      ns("demographics_file"),
      "Demographics File",
      choices = c("default" = "demographic_factors.csv")
    ),
    shiny::textInput(ns("scenario_name"), "Scenario Name"),
    shinyjs::disabled(
      shiny::fileInput(ns("params_upload"), "Upload Params Excel File", accept = ".xlsx")
    ),
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

    status <- reactiveVal("waiting")

    observe({
      valid <- c("dataset", "demographics_file", "scenario_name") |>
        purrr::every(~ shiny::isTruthy(input[[.x]]))
      shinyjs::toggleState("params_upload", valid)
    })

    params <- reactive({
      dataset <- input$dataset
      demographics_file <- input$demographics_file
      scenario_name <- input$scenario_name

      file <- shiny::req(input$params_upload)
      ext <- tools::file_ext(file$datapath)
      shiny::validate(
        shiny::need(ext == "xlsx", "Please upload an xlsx file"),
        shiny::need(shiny::isTruthy(dataset), "Select a dataset"),
        shiny::need(shiny::isTruthy(demographics_file), "Select a demographics file"),
        shiny::need(shiny::isTruthy(scenario_name), "Enter a scenario name")
      )

      status("processing file...")
      on.exit({
        status("file loaded, submitting to batch")
      })
      process_param_file(file$datapath, dataset, demographics_file, scenario_name)
    }) |>
      bindEvent(input$params_upload) |>
      debounce(1000)

    job_name <- shiny::observeEvent(params(), {
      params <- req(params())

      job_name <- add_job(params)

      status(paste("submitted to batch:", job_name))
    })
    output$status <- shiny::renderText({
      status()
    })
  })
}
