#' The application server-side
#'
#' @param input,output,session Internal parameters for {shiny}.
#'     DO NOT REMOVE.
#' @noRd
app_server <- function(input, output, session) {

  # this should probably become parameterised
  data_path <- reactive({
    "../data/RL4/results/test/20220110_104353"
  })

  data <- reactive({
    arrow::read_parquet(glue::glue("{data_path()}/model_results.parquet")) |>
      dplyr::as_tibble() |>
      dplyr::select(-.data$`__index_level_0__`)
  })

  mod_principal_high_level_server("principal_high_level", data)
  mod_principal_detailed_server("principal_detailed", data)
}