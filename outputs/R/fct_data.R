#' data
#'
#' @description A fct function
#'
#' @return The return value, if any, from executing the function.
#'
#' @noRd
get_data <- function(p) {
  selected_variants <- jsonlite::read_json(file.path(p, "run_params.json"))[["variant"]] |>
    purrr::flatten_chr() |>
    tibble::enframe("model_run", "variant") |>
    dplyr::mutate(dplyr::across(.data$model_run, `-`, 1)) |> # R uses 1 based indexing, so subtract 1
    arrow::to_duckdb()

  arrow::open_dataset(file.path(p, "aggregated_results")) |>
    arrow::to_duckdb() |>
    dplyr::left_join(selected_variants, by = "model_run")
}

get_change_factors <- function(p) {
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
}