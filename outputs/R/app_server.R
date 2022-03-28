#' The application server-side
#'
#' @param input,output,session Internal parameters for {shiny}.
#'     DO NOT REMOVE.
#' @noRd
app_server <- function( input, output, session ) {

  data <- reactive({
    # trigger on load, need to change to be based on some dropdowns?
    data_path <- "../data/RL4/results/test/20220110_104353"

    list(
      aae = arrow::read_parquet(glue::glue("{data_path}/aae_principal.parquet")) |>
        rename(pod = aedepttype),
      ip  = arrow::read_parquet(glue::glue("{data_path}/ip_principal.parquet")),
      op  = arrow::read_parquet(glue::glue("{data_path}/op_principal.parquet"))
    )
  })

  mod_principal_high_level_server("principal_high_level", data)
  mod_principal_detailed_server("principal_detailed", data)
}
