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
    shiny::h1("Core change factor effects"),
    shiny::h2("Inpatient spell estimated impact (principal projection)"),
    shinycssloaders::withSpinner(
      plotly::plotlyOutput(ns("change_factors"), height = "600px")
    )
  )
}

#' principal_change_factor_effects Server Functions
#'
#' @noRd
mod_principal_change_factor_effects_server <- function(id, change_factors) {
  moduleServer(id, function(input, output, session) {
    change_factors_summarised <- reactive({
      change_factors() |>
        dplyr::filter(.data$dataset == "ip", .data$model_run == 0, .data$measure == "Admissions") |>
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
        dplyr::select(-.data$cuvalue) |>
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
            rev(c(levels(change_factors()$change_factor), "Estimate"))
          )
        )

      p <- ggplot2::ggplot(d, aes(.data$value, .data$change_factor)) +
        ggplot2::geom_col(aes(fill = .data$colour), show.legend = FALSE, position = "stack") +
        ggplot2::scale_fill_identity()

      plotly::ggplotly(p) |>
        plotly::layout(showlegend = FALSE)
    })
  })
}