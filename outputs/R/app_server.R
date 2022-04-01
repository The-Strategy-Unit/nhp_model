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
    arrow::read_parquet(file.path(p, "model_results.parquet")) |>
      dplyr::as_tibble() |>
      dplyr::mutate(
        dplyr::across(
          .data$age_group,
          forcats::fct_relevel,
          "0-4",
          "5-14",
          "15-34",
          "35-49",
          "50-64",
          "65-84",
          "85+"
        )
      )
  })
  change_factors <- reactive({
    p <- shiny::req(data_path())
    readr::read_csv(file.path(p, "change_factors.csv"), col_types = "ccddcd") |>
      dplyr::select(
        .data$dataset,
        .data$change_factor,
        .data$type,
        .data$model_run,
        .data$rows,
        .data$beddays
      ) |>
      dplyr::mutate(
        dplyr::across(
          c(.data$change_factor, .data$type),
          forcats::fct_inorder
        ),
        dplyr::across(
          c(.data$change_factor, .data$type),
          forcats::fct_relabel,
          snakecase::to_title_case
        ),
        dplyr::across(.data$type, forcats::fct_recode, "NULL Strategy" = "Null")
      )
  })

  mod_principal_high_level_server("principal_high_level", data)
  mod_principal_detailed_server("principal_detailed", data)
  mod_principal_change_factor_effects_server("principal_change_factor_effects", change_factors)

  mod_model_core_activity_server("model_core_activity", data)
  mod_model_results_distribution_server("model_results_distribution", data)
}