
get_activity_type_pod_measure_options <- function() {
  get_golem_config("pod_measures") |>
    purrr::map_dfr(\(.x) {
      .x$pods |>
        purrr:::map_dfr(tibble::as_tibble, .id = "pod") |>
        dplyr::transmute(
          activity_type_name = .x$name,
          .data$pod,
          pod_name = .data$name,
          .data$measures
        )
    }, .id = "activity_type")
}
