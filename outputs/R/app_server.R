#' The application server-side
#'
#' @param input,output,session Internal parameters for {shiny}.
#'     DO NOT REMOVE.
#' @noRd
app_server <- function(input, output, session) {

  # this module returns a reactive which contains the data path
  data_path <- mod_result_selection_server("result_selection")
  data <- reactive({
    get_data(shiny::req(data_path()))
  })
  change_factors <- reactive({
    get_change_factors(shiny::req(data_path()))
  })

  mod_params_upload_server("params_upload_ui")

  mod_principal_high_level_server("principal_high_level", data)
  mod_principal_detailed_server("principal_detailed", data)
  mod_principal_change_factor_effects_server("principal_change_factor_effects", change_factors)

  mod_model_core_activity_server("model_core_activity", data)
  mod_model_results_distribution_server("model_results_distribution", data)
}