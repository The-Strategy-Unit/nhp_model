
cosmos_get_container <- function(container) {
  endp <- AzureCosmosR::cosmos_endpoint(
    Sys.getenv("COSMOS_ENDPOINT"),
    Sys.getenv("COSMOS_KEY")
  )

  db <- AzureCosmosR::get_cosmos_database(endp, "nhp_results")

  AzureCosmosR::get_cosmos_container(db, container)
}

cosmos_get_datasets <- function() {
  container <- cosmos_get_container("results")

  qry <- glue::glue("
SELECT DISTINCT
    c.dataset
FROM c
WHERE
    c.model_run = 0
")
  AzureCosmosR::query_documents(container, qry, cross_partion = FALSE, partition_key = 0) |>
    purrr::pluck("dataset") |>
    sort()
}

cosmos_get_scenarios <- function(ds) {
  container <- cosmos_get_container("results")

  qry <- glue::glue("
SELECT DISTINCT
    c.scenario
FROM c
WHERE
    c.model_run = 0
AND
    c.dataset = '{ds}'
")
  AzureCosmosR::query_documents(container, qry, cross_partion = FALSE, partition_key = 0) |>
    purrr::pluck("scenario") |>
    sort()
}

cosmos_get_create_datetimes <- function(ds, sc) {
  container <- cosmos_get_container("results")

  qry <- glue::glue("
SELECT DISTINCT
    c.create_datetime
FROM c
WHERE
    c.model_run = 0
AND
    c.dataset = '{ds}'
AND
    c.scenario = '{sc}'
")
  AzureCosmosR::query_documents(container, qry, cross_partion = FALSE, partition_key = 0) |>
    purrr::pluck("create_datetime") |>
    sort()
}

cosmos_get_params <- function(ds, sc, cd) {
  container <- cosmos_get_container("params")

  id <- glue::glue("{ds}|{sc}|{cd}")

  AzureCosmosR::get_document(container, id, id)$data
}

cosmos_get_principal_highlevel <- function(ds, sc, cd) {
  container <- cosmos_get_container("results")

  qry <- glue::glue("
SELECT
    c.model_run,
    r.pod,
    r[\"value\"]
FROM c
JOIN r IN c.results[\"default\"]
WHERE
    c.model_run <= 0
AND
    c.id LIKE '{ds}|{sc}|{cd}|%'
AND
    r.measure IN ('admissions', 'attendances', 'procedures', 'ambulance', 'walk-in')
")

  AzureCosmosR::query_documents(container, qry) |>
    dplyr::as_tibble() |>
    dplyr::mutate(
      dplyr::across(.data$model_run, model_run_type),
      dplyr::across(.data$pod, ~ ifelse(stringr::str_starts(.x, "aae"), "aae", .x))
    ) |>
    dplyr::count(.data$model_run, .data$pod, wt = .data$value)
}

cosmos_get_model_core_activity <- function(ds, sc, cd) {
  container <- cosmos_get_container("results")

  qry <- glue::glue("
SELECT
    c.model_run,
    r.pod,
    r.measure,
    r[\"value\"]
FROM c
JOIN r IN c.results[\"default\"]
WHERE
    c.model_run != 0
AND
    c.id LIKE '{ds}|{sc}|{cd}|%'
")

  AzureCosmosR::query_documents(container, qry) |>
    dplyr::as_tibble()
}

cosmos_get_aggregation <- function(ds, sc, cd, pod, measure, agg_col) {
  container <- cosmos_get_container("results")

  agg_type <- glue::glue("sex+{agg_col}")
  qry <- glue::glue("
SELECT
    c.model_run,
    c.selected_variant as variant,
    r.sex,
    r.{agg_col},
    r[\"value\"]
FROM c
JOIN r IN c.results[\"{agg_type}\"]
WHERE
    c.id LIKE '{ds}|{sc}|{cd}|%'
AND
    r.pod = '{pod}'
AND
    r.measure = '{measure}'
")

  AzureCosmosR::query_documents(container, qry) |>
    dplyr::as_tibble() |>
    dplyr::mutate(type = model_run_type(.data$model_run))
}

cosmos_get_principal_change_factors <- function(ds, sc, cd, activity_type) {
  container <- cosmos_get_container("results")

  qry <- glue::glue("
SELECT
    r.change_factor,
    r.strategy,
    r.measure,
    r[\"value\"]
FROM c
JOIN r IN c.change_factors
WHERE
    c.id LIKE '{ds}|{sc}|{cd}|{activity_type}'
AND
    c.model_run = 0
")

  AzureCosmosR::query_documents(container, qry) |>
    dplyr::as_tibble()
}