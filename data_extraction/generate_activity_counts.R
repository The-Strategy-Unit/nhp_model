generate_activity_counts <- function(ip_data, op_data, aae_data) {
  .data <- rlang::.data
  c(
    stringr::str_subset(ip_data, "ip\\.parquet$"),
    op_data,
    aae_data
  ) |>
    purrr::set_names() |>
    purrr::map(\(.x) {
      .x |>
        arrow::read_parquet() |>
        dplyr::count(.data[["group"]])
    }) |>
    purrr::discard(\(.x) nrow(.x) == 0) |>
    dplyr::bind_rows(.id = "file") |>
    dplyr::mutate(
      dplyr::across(
        "file",
        \(.x) stringr::str_remove_all(.x, "data/|\\.parquet")
      )
    ) |>
    tidyr::separate("file", c("year", "provider", "activity_type"), "/")
}
