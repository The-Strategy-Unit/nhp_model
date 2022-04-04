#' The application server-side
#'
#' @param input,output,session Internal parameters for {shiny}.
#'     DO NOT REMOVE.
#' @noRd
app_server <- function(input, output, session) {

  # this module returns a reactive which contains the data path
  data_path <- mod_result_selection_server("result_selection")
  data <- reactive({
    p <- shiny::req(data_path())
    arrow::open_dataset(file.path(p, "aggregated_results")) |>
      arrow::to_duckdb()
  })
  change_factors <- reactive({
    p <- shiny::req(data_path())
    arrow::open_dataset(file.path(p, "change_factors"), format = "csv") |>
      dplyr::collect() |>
      dplyr::mutate(
        dplyr::across(
          c(.data$change_factor, .data$strategy),
          forcats::fct_inorder
        ),
        dplyr::across(
          c(.data$change_factor, .data$strategy, .data$measure),
          forcats::fct_relabel,
          snakecase::to_title_case
        ),
        dplyr::across(.data$strategy, forcats::fct_recode, "NULL Strategy" = "Null")
      )
  })

  mod_principal_high_level_server("principal_high_level", data)
  mod_principal_detailed_server("principal_detailed", data)
  mod_principal_change_factor_effects_server("principal_change_factor_effects", change_factors)

  mod_model_core_activity_server("model_core_activity", data)
  mod_model_results_distribution_server("model_results_distribution", data)
}