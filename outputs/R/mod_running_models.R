#' running_models UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_running_models_ui <- function(id){
  ns <- NS(id)
  tagList(
    h2("Running Models"),
    tableOutput(ns("running_models"))
  )
}

#' running_models Server Functions
#'
#' @noRd
mod_running_models_server <- function(id){
  moduleServer( id, function(input, output, session){
    refresh_timer <- reactiveTimer(2500)

    output$running_models <- renderTable({
      job_status <- function(job_id) {
        get_tasks(job_id) |>
          dplyr::summarise(
            complete = sum(.data$state == "completed"),
            running = sum(.data$state == "running"),
            n = n()
          )
      }

      get_jobs() |>
        dplyr::filter(.data$state != "completed") |>
        dplyr::mutate(status = purrr::map_dfr(.data$id, job_status)) |>
        tidyr::unnest(status)
    }) |>
      bindEvent(refresh_timer())
  })
}
