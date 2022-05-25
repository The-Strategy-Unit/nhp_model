#' principal_detailed UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_principal_detailed_ui <- function(id) {
  ns <- NS(id)
  tagList(
    shiny::h1("Detailed activity estimates (principal projection)"),
    shiny::fluidRow(
      mod_measure_selection_ui(ns("measure_selection"), width = 3),
      col_3(
        selectInput(
          ns("aggregation"),
          "Aggregation",
          # TODO: this should be taken from golem-config.yml
          c("Age Group", "Treatment Specialty")
        )
      )
    ),
    shinycssloaders::withSpinner(
      gt::gt_output(ns("results"))
    )
  )
}

#' principal_detailed Server Functions
#'
#' @noRd
mod_principal_detailed_server <- function(id, selected_model_run, data_cache) {
  moduleServer(id, function(input, output, session) {
    selected_measure <- mod_measure_selection_server("measure_selection")

    selected_data <- reactive({
      id <- selected_model_run()
      c(pod, measure) %<-% selected_measure()

      agg_col <- switch(
        req(input$aggregation),
        "Age Group" = "age_group",
        "Treatment Specialty" = "tretspef"
      )

      cosmos_get_aggregation(id, pod, measure, agg_col) |>
        dplyr::transmute(
          .data$sex,
          agg = .data[[agg_col]],
          .data$baseline,
          final = .data$principal,
          change = final - baseline, change_pcnt = change / baseline
        )
    }) |>
      shiny::bindCache(selected_model_run(), selected_measure(), cache = data_cache)

    output$results <- gt::render_gt({
      d <- selected_data()

      # handle some edge cases where a dropdown is changed and the next dropdowns aren't yet changed: we get 0 rows of
      # data which causes a bunch of warning messages
      req(nrow(d) > 0)

      d |>
        dplyr::mutate(
          dplyr::across(.data$sex, ~ ifelse(.x == 1, "Male", "Female")),
          dplyr::across(.data$final, gt_bar, scales::comma_format(1), "#686f73", "#686f73"),
          dplyr::across(.data$change, gt_bar, scales::comma_format(1)),
          dplyr::across(.data$change_pcnt, gt_bar, scales::percent_format(1))
        ) |>
        gt::gt(groupname_col = "sex") |>
        gt::cols_label(
          agg = req(input$aggregation),
          baseline = "Baseline",
          final = "Final",
          change = "Change",
          change_pcnt = "Percent Change",
        ) |>
        gt::fmt_integer(c(baseline)) |>
        gt::cols_width(final ~ px(150), change ~ px(150), change_pcnt ~ px(150)) |>
        gt::cols_align(
          align = "left",
          columns = c("agg", "final", "change", "change_pcnt")
        ) |>
        gt_theme()
    })
  })
}
