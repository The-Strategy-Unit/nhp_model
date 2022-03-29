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
    h1("Detailed activity estimates (principal projection)"),
    mod_measure_selection_ui(ns("measure_selection")),
    shinycssloaders::withSpinner(
      gt::gt_output(ns("results"))
    )
  )
}

#' principal_detailed Server Functions
#'
#' @noRd
mod_principal_detailed_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    filtered_data <- mod_measure_selection_server("measure_selection", data)

    selected_data <- reactive({
      d <- filtered_data()
      req(nrow(d) > 0)

      d |>
        dplyr::filter(.data$type != "model") |>
        dplyr::select(-.data$model_run) |>
        tidyr::pivot_wider(names_from = .data$type, values_from = .data$value, values_fill = 0) |>
        dplyr::rename(final = .data$principal) |>
        dplyr::mutate(change = final - baseline, change_pcnt = change / baseline)
    })

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
          agg = attr(filtered_data, "aggregation"),
          baseline = "Baseline",
          final = "Final",
          change = "Change",
          change_pcnt = "Percent Change",
        ) |>
        gt::fmt_integer(c(baseline)) |>
        gt::cols_width(final ~ px(150), change ~ px(150), change_pcnt ~ px(150)) |>
        gt::cols_align(
          align = "left" # ,
          # columns = c("age_group", "final", "change", "change_pcnt")
        )
    })

    observeEvent(input$activity_type, {
      at <- req(input$activity_type)

      p <- dropdown_options() |>
        dplyr::filter(.data$dataset == at) |>
        dplyr::pull(.data$pod) |>
        unique()

      shiny::updateSelectInput(session, "pod", choices = p)
    })

    observeEvent(input$pod, {
      at <- req(input$activity_type)
      p <- req(input$pod)

      m <- dropdown_options() |>
        dplyr::filter(.data$dataset == at, .data$pod == p) |>
        dplyr::pull(.data$measure) |>
        unique()

      shiny::updateSelectInput(session, "measure", choices = m)
    })
  })
}