#' grouped_complete
#'
#' @description A fct function
#'
#' @return The return value, if any, from executing the function.
#'
#' @noRd
grouped_complete <- function(data, ..., fill = list(), explicit = TRUE) {
  groups <- dplyr::groups(data)
  data |>
    dplyr::ungroup() |>
    tidyr::complete(..., fill = fill, explicit = explicit) |>
    dplyr::group_by(!!!groups)
}