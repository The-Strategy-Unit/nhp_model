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
mod_principal_change_factor_effects_server <- function(id, change_factors) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    observe({
      d <- change_factors() |>
        distinct(.data$activity_type)
      req(nrow(d) > 0)

      activity_types <- activity_type_display |>
        dplyr::semi_join(d, by = "activity_type") |>
        (function(.x) purrr::set_names(.x[[1]], .x[[2]]))()

      shiny::updateSelectInput(session, "activity_type", choices = activity_types)
    })

    observeEvent(input$activity_type, {
      at <- req(input$activity_type)

      measures <- change_factors() |>
        filter(.data$activity_type == at) |>
        pull(.data$measure) |>
        unique()
      req(length(measures) > 0)

      shiny::updateSelectInput(session, "measure", choices = measures)
    })

    principal_change_factors <- reactive({
      at <- req(input$activity_type)
      m <- req(input$measure)

      change_factors() |>
        dplyr::filter(
          .data$activity_type == at,
          .data$model_run == 0,
          .data$measure == m
        )
    })

    change_factors_summarised <- reactive({
      principal_change_factors() |>
        dplyr::filter(
          input$include_baseline | .data$change_factor != "Baseline"
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
      d <- principal_change_factors() |>
        dplyr::filter(
          .data$strategy != "",
          .data$value < 0
        )

      if (input$sort_type == "descending value") {
        d <- dplyr::mutate(d, dplyr::across(.data$strategy, forcats::fct_reorder, -.data$value))
      }

      d
    })

    observeEvent(individual_change_factors(), {
      d <- individual_change_factors()

      shinyjs::toggle("individial_change_factors", condition = nrow(d) > 0)
    })

    output$admission_avoidance <- plotly::renderPlotly({
      individual_change_factors() |>
        dplyr::filter(.data$change_factor == "Admission Avoidance") |>
        ggplot2::ggplot(aes(.data$value, .data$strategy)) +
        ggplot2::geom_col(fill = "#f9bf07") +
        ggplot2::scale_x_continuous(labels = scales::comma) +
        labs(x = "", y = "")
    })

    output$los_reduction <- plotly::renderPlotly({
      individual_change_factors() |>
        dplyr::filter(.data$change_factor == "Los Reduction") |>
        ggplot2::ggplot(aes(.data$value, .data$strategy)) +
        ggplot2::geom_col(fill = "#ec6555") +
        ggplot2::scale_x_continuous(labels = scales::comma) +
        labs(x = "", y = "")
    })
  })
}
