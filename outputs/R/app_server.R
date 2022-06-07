#' The application server-side
#'
#' @param input,output,session Internal parameters for {shiny}.
#'     DO NOT REMOVE.
#' @noRd
app_server <- function(input, output, session) {

  # create a 200 MiB cache on disk
  data_cache <- cachem::cache_disk(dir = ".cache/data_cache", max_size = 200 * 1024^2)

  # in case we need to invalidate the cache on rsconnect quickly, we can increment the "CACHE_VERSION" env var
  cache_version <- ifelse(
    file.exists(".cache/cache_version.txt"),
    as.numeric(readLines(".cache/cache_version.txt")),
    -1
  )

  if (Sys.getenv("CACHE_VERSION", 0) > cache_version) {
    cat("Invalidating cache\n")
    data_cache$reset()
    cache_version <- Sys.getenv("CACHE_VERSION", 0)
    writeLines(as.character(cache_version), ".cache/cache_version.txt")
  }

  # this module returns a reactive which contains the data path
  selected_model_run <- mod_result_selection_server("result_selection")
  selected_model_run_id <- reactive({
    selected_model_run()$id
  })

  mod_params_upload_server("params_upload_ui")
  mod_running_models_server("running_models")

  mod_principal_high_level_server("principal_high_level", selected_model_run_id, data_cache)
  mod_principal_detailed_server("principal_detailed", selected_model_run_id, data_cache)
  mod_principal_change_factor_effects_server("principal_change_factor_effects", selected_model_run_id, data_cache)

  mod_model_core_activity_server("model_core_activity", selected_model_run_id, data_cache)

}
