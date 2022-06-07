#' principal_change_factor_effects UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_principal_change_factor_effects_ui <- function(id) {
  ns <- shiny::NS(id)
  shiny::tagList(
    shiny::h1("Core change factor effects (principal projection)"),
    shiny::fluidRow(
      col_4(shiny::selectInput(ns("activity_type"), "Activity Type", NULL)),
      col_4(shiny::selectInput(ns("measure"), "Measure", NULL)),
      shiny::checkboxInput(ns("include_baseline"), "Include baseline?", TRUE)
    ),
    shinycssloaders::withSpinner(
      plotly::plotlyOutput(ns("change_factors"), height = "600px")
    ),
    shinyjs::hidden(
      shiny::tags$div(
        id = ns("individial_change_factors"),
        shiny::h2("Individual Change Factors"),
        shiny::selectInput(ns("sort_type"), "Sort By", c("alphabetical", "descending value")),
        shinycssloaders::withSpinner(
          fluidRow(
            col_6(plotly::plotlyOutput(ns("admission_avoidance"), height = "600px")),
            col_6(plotly::plotlyOutput(ns("los_reduction"), height = "600px"))
          )
        )
      )
    )
  )
}

#' principal_change_factor_effects Server Functions
#'
#' @noRd
mod_principal_change_factor_effects_server <- function(id, selected_model_run_id, data_cache) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    observe({
      activity_types <- get_activity_type_pod_measure_options() |>
        dplyr::distinct(
          dplyr::across(
            tidyselect::starts_with("activity_type")
          )
        ) |>
        set_names()

      shiny::updateSelectInput(session, "activity_type", choices = activity_types)
    })

    principal_change_factors <- reactive({
      at <- req(input$activity_type)

      id <- selected_model_run_id()

      cosmos_get_principal_change_factors(id, at) |>
        dplyr::mutate(
          dplyr::across(.data$change_factor, forcats::fct_inorder),
          dplyr::across(
            .data$change_factor,
            forcats::fct_relevel,
            "baseline",
            "population_factors",
            "health_status_adjustment"
          )
        )
    }) |>
      shiny::bindCache(selected_model_run_id(), input$activity_type, cache = data_cache)

    observeEvent(principal_change_factors(), {
      at <- req(input$activity_type)
      pcf <- req(principal_change_factors())

      measures <- unique(pcf$measure)

      req(length(measures) > 0)

      shiny::updateSelectInput(session, "measure", choices = measures)
    })

    change_factors_summarised <- reactive({
      m <- req(input$measure)

      principal_change_factors() |>
        dplyr::filter(
          .data$measure == m,
          input$include_baseline | .data$change_factor != "baseline"
        ) |>
        dplyr::group_by(.data$change_factor) |>
        dplyr::summarise(dplyr::across(.data$value, sum, na.rm = TRUE)) |>
        dplyr::mutate(cuvalue = cumsum(.data$value))
    })

    output$change_factors <- plotly::renderPlotly({
      d <- change_factors_summarised() |>
        dplyr::mutate(
          hidden = tidyr::replace_na(lag(.data$cuvalue) + pmin(.data$value, 0), 0),
          colour = case_when(
            .data$change_factor == "Baseline" ~ "#686f73",
            .data$value >= 0 ~ "#f9bf07",
            TRUE ~ "#2c2825"
          ),
          across(.data$value, abs)
        ) |>
        dplyr::select(-.data$cuvalue)

      levels <- c(levels(forcats::fct_drop(d$change_factor)), "Estimate")

      d <- d |>
        dplyr::bind_rows(
          dplyr::tibble(
            change_factor = "Estimate",
            value = sum(change_factors_summarised()$value),
            hidden = 0,
            colour = "#ec6555"
          )
        ) |>
        tidyr::pivot_longer(c(.data$value, .data$hidden)) |>
        dplyr::mutate(
          across(.data$colour, ~ ifelse(.data$name == "hidden", NA, .x)),
          across(.data$name, forcats::fct_relevel, "hidden", "value"),
          across(
            .data$change_factor,
            forcats::fct_relevel,
            rev(levels)
          )
        )

      p <- ggplot2::ggplot(d, aes(.data$value, .data$change_factor)) +
        ggplot2::geom_col(aes(fill = .data$colour), show.legend = FALSE, position = "stack") +
        ggplot2::scale_fill_identity() +
        ggplot2::scale_x_continuous(labels = scales::comma)

      plotly::ggplotly(p) |>
        plotly::layout(showlegend = FALSE)
    })

    individual_change_factors <- reactive({
      m <- req(input$measure)

      d <- principal_change_factors() |>
        dplyr::filter(
          .data$measure == m,
          .data$strategy != "-",
          .data$value < 0
        )

      if (input$sort_type == "descending value") {
        dplyr::mutate(d, dplyr::across(.data$strategy, forcats::fct_reorder, -.data$value))
      } else {
        dplyr::mutate(d, dplyr::across(.data$strategy, forcats::fct_reorder, .data$strategy))
      }
    })

    observeEvent(individual_change_factors(), {
      d <- individual_change_factors()

      shinyjs::toggle("individial_change_factors", condition = nrow(d) > 0)
    })

    output$admission_avoidance <- plotly::renderPlotly({
      individual_change_factors() |>
        dplyr::filter(.data$change_factor == "admission_avoidance") |>
        ggplot2::ggplot(aes(.data$value, .data$strategy)) +
        ggplot2::geom_col(fill = "#f9bf07") +
        ggplot2::scale_x_continuous(labels = scales::comma) +
        labs(x = "", y = "")
    })

    output$los_reduction <- plotly::renderPlotly({
      individual_change_factors() |>
        dplyr::filter(.data$change_factor == "los_reduction") |>
        ggplot2::ggplot(aes(.data$value, .data$strategy)) +
        ggplot2::geom_col(fill = "#ec6555") +
        ggplot2::scale_x_continuous(labels = scales::comma) +
        labs(x = "", y = "")
    })
  })
}