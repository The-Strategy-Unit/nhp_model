#' data
#'
#' @description A fct function
#'
#' @return The return value, if any, from executing the function.
#'
#' @noRd
get_data <- function(db_con, ds, sc, cd) {
  selected_variants_query <- glue::glue_sql(.con = db_con, "SELECT * FROM selected_variant({ds}, {sc}, {cd})")
  selected_variants <- DBI::dbGetQuery(db_con, selected_variants_query)

  dplyr::tbl(db_con, "aggregated_results") |>
    dplyr::filter(.data$dataset == ds, .data$scenario == sc, .data$create_datetime == cd) |>
    dplyr::select(-.data$dataset, -.data$scenario, -.data$create_datetime) |>
    dplyr::collect() |>
    dplyr::left_join(selected_variants, by = "model_run")
}

get_change_factors <- function(db_con, ds, sc, cd) {
  dplyr::tbl(db_con, "change_factors") |>
    dplyr::filter(.data$dataset == ds, .data$scenario == sc, .data$create_datetime == cd) |>
    dplyr::select(-.data$dataset, -.data$scenario, -.data$create_datetime) |>
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
}
