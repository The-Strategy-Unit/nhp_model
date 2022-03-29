#' The application server-side
#'
#' @param input,output,session Internal parameters for {shiny}.
#'     DO NOT REMOVE.
#' @noRd
app_server <- function(input, output, session) {

  # this module returns a reactive which contains the loaded data
  data <- mod_result_selection_server("result_selection")

  mod_principal_high_level_server("principal_high_level", data)
  mod_principal_detailed_server("principal_detailed", data)
  mod_model_core_activity_server("model_core_activity", data)
  mod_model_results_distribution_server("model_results_distribution", data)
}