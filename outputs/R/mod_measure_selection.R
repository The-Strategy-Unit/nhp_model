#' measure_selection UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_measure_selection_ui <- function(id, width = 4) {
  ns <- NS(id)

  tagList(
    column(width, selectInput(ns("activity_type"), "Activity Type", NULL)),
    column(width, selectInput(ns("pod"), "POD", NULL)),
    column(width, selectInput(ns("measure"), "Measure", NULL))
  )
}

#' measure_selection Server Functions
#'
#' @noRd
mod_measure_selection_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    atpmo <- get_activity_type_pod_measure_options()

    # handle onload
    observe({
      activity_types <- atpmo |>
        dplyr::distinct(
          dplyr::across(
            tidyselect::starts_with("activity_type")
          )
        ) |>
        set_names()

      shiny::updateSelectInput(session, "activity_type", choices = activity_types)
    })

    shiny::observeEvent(input$activity_type, {
      at <- req(input$activity_type)

      pods <- atpmo |>
        dplyr::filter(.data$activity_type == at) |>
        dplyr::distinct(
          dplyr::across(
            tidyselect::starts_with("pod")
          )
        ) |>
        set_names()

      shiny::updateSelectInput(session, "pod", choices = pods)
    })

    shiny::observeEvent(input$pod, {
      at <- req(input$activity_type)
      p <- req(input$pod)

      # TODO: could there be a better way of handling this?
      aggregations <- c("Age Group", "Treatment Specialty")
      if (at == "aae") {
        shiny::updateSelectInput(session, "aggregation", choices = aggregations[1])
      } else {
        shiny::updateSelectInput(session, "aggregation", choices = aggregations)
      }

      measures <- atpmo |>
        dplyr::filter(.data$activity_type == at, .data$pod == p) |>
        purrr::pluck("measures")

      shiny::updateSelectInput(session, "measure", choices = measures)
    })

    selected_measure <- reactive({
      p <- req(input$pod)
      m <- req(input$measure)

      # ensure a valid set of pod/measure has been selected. If activity type changes we may end up with invalid options
      req(nrow(dplyr::filter(atpmo, .data$pod == p, .data$measures == m)) > 0)

      c(pod = p, measure = m)
    })
    return(selected_measure)
  })
}
