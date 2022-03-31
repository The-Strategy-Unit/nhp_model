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
  ns <- NS(id)
  tagList(
    h1("Core change factor effects"),
    h2("Inpatient spell estimated impact (principal projection)"),
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
    ns <- session$ns

    change_factors_summarised <- reactive({
      change_factors() |>
        dplyr::group_by(.data$type) |>
        dplyr::summarise(dplyr::across(.data$value, sum)) |>
        dplyr::mutate(cuvalue = cumsum(.data$value))
    })

    output$change_factors <- plotly::renderPlotly({
      d <- change_factors_summarised() |>
        dplyr::mutate(
          hidden = tidyr::replace_na(lag(.data$cuvalue) + pmin(.data$value, 0), 0),
          colour = case_when(
            .data$type == "Baseline" ~ "#686f73",
            .data$value >= 0 ~ "#f9bf07",
            TRUE ~ "#2c2825"
          ),
          across(.data$value, abs)
        ) |>
        dplyr::select(-.data$cuvalue) |>
        dplyr::bind_rows(
          dplyr::tibble(
            type = "Estimate",
            value = sum(change_factors_summarised()$value),
            hidden = 0,
            colour = "#ec6555"
          )
        ) |>
        tidyr::pivot_longer(c(.data$value, .data$hidden)) |>
        dplyr::mutate(
          across(.data$colour, ~ ifelse(.data$name == "hidden", NA, .x)),
          across(.data$type, forcats::fct_inorder),
          across(.data$name, forcats::fct_relevel, "hidden", "value")
        )

      p <- ggplot2::ggplot(d, aes(.data$value, .data$type)) +
        ggplot2::geom_col(aes(fill = .data$colour), show.legend = FALSE, position = "stack") +
        ggplot2::scale_fill_identity()

      plotly::ggplotly(p) |>
        plotly::layout(showlegend = FALSE)
    })
  })
}
