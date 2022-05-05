#' data
#'
#' @description A fct function
#'
#' @return The return value, if any, from executing the function.
#'
#' @noRd
get_data <- function(db_con, ds, sc, mr) {
  dplyr::tbl(db_con, "aggregated_results") |>
    dplyr::filter(.data$dataset == ds, .data$scenario == sc, .data$create_datetime == mr) |>
    dplyr::select(-.data$dataset, -.data$scenario, -.data$create_datetime) |>
    dplyr::collect() |>
    dplyr::mutate(variant = "TODO", dplyr::across(.data$value, as.numeric))
}

get_change_factors <- function(db_con, ds, sc, mr) {
  dplyr::tbl(db_con, "change_factors") |>
    dplyr::filter(.data$dataset == ds, .data$scenario == sc, .data$create_datetime == mr) |>
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
