#' measure_selection UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_measure_selection_ui <- function(id, aggregation = TRUE) {
  ns <- NS(id)

  width <- if (aggregation) 3 else 4

  tagList(
    fluidRow(
      column(width, selectInput(ns("activity_type"), "Activity Type", NULL)),
      column(width, selectInput(ns("pod"), "POD", NULL)),
      column(width, selectInput(ns("measure"), "Measure", NULL)),
      # no matter what, we need to include this input, but we can hide it with CSS
      column(
        width,
        style = glue::glue("display: {if (aggregation) 'block' else 'none'}"),
        selectInput(
          ns("aggregation"),
          "Aggregation",
          # TODO: this should be taken from golem-config.yml
          c("Age Group", "Treatment Specialty")
        )
      )
    )
  )
}

#' measure_selection Server Functions
#'
#' @noRd
mod_measure_selection_server <- function(id, selected_model_run, data_cache) {
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

      measures <- atpmo |>
        dplyr::filter(.data$activity_type == at, .data$pod == p) |>
        purrr::pluck("measures")

      shiny::updateSelectInput(session, "measure", choices = measures)
    })

    filtered_data <- reactive({
      c(ds, sc, cd) %<-% selected_model_run()

      p <- req(input$pod)
      m <- req(input$measure)

      # ensure a valid set of pod/measure has been selected. If activity type changes we may end up with invalid options
      req(nrow(dplyr::filter(atpmo, .data$pod == p, .data$measures == m)) > 0)

      agg_col <- switch(req(input$aggregation),
        "Age Group" = "age_group",
        "Treatment Specialty" = "tretspef"
      )

      d <- cosmos_get_aggregation(ds, sc, cd, p, m, agg_col) |>
        dplyr::rename(agg = tidyselect::all_of(agg_col))

      attr(d, "aggregation") <- input$aggregation

      return(d)
    }) |>
      shiny::bindCache(selected_model_run(), input$pod, input$measure, input$aggregation, cache = data_cache)

    return(filtered_data)
  })
}
