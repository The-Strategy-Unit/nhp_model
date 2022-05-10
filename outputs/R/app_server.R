#' The application server-side
#'
#' @param input,output,session Internal parameters for {shiny}.
#'     DO NOT REMOVE.
#' @noRd
app_server <- function(input, output, session) {

  # this module returns a reactive which contains the data path
  selected_model_run <- mod_result_selection_server("result_selection")
  data <- reactive({
    selected_model_run()$data
  })
  change_factors <- reactive({
    selected_model_run()$change_factors
  })
  years <- reactive({
    selected_model_run()$years
  })

  mod_params_upload_server("params_upload_ui")

  mod_principal_high_level_server("principal_high_level", data, years)
  mod_principal_detailed_server("principal_detailed", data)
  mod_principal_change_factor_effects_server("principal_change_factor_effects", change_factors)

  mod_model_core_activity_server("model_core_activity", data)
  mod_model_results_distribution_server("model_results_distribution", data)
}