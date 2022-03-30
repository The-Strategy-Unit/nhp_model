#' gt_theme
#'
#' @description A fct function
#'
#' @return The return value, if any, from executing the function.
#'
#' @noRd
gt_theme <- function(data) {
  gt::tab_options(
    data = data,
    heading.subtitle.font.size = 12,
    heading.align = "left",
    row_group.border.top.width = gt::px(2),
    row_group.border.top.color = "black",
    row_group.border.bottom.color = "black",
    row_group.background.color = "#fcdf83",
    table_body.hlines.color = "white",
    table.border.top.color = "white",
    table.border.top.width = gt::px(2),
    table.border.bottom.color = "white",
    table.border.bottom.width = gt::px(3),
    column_labels.border.bottom.color = "black",
    column_labels.border.bottom.width = gt::px(1)
  )
}