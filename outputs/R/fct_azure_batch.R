batch_token_fn <- function(resource) {
  suppressMessages({
    if (Sys.getenv("AAD_TENANT_ID") == "") {
      AzureAuth::get_managed_token(resource)
    } else {
      AzureAuth::get_azure_token(
        resource,
        Sys.getenv("AAD_TENANT_ID"),
        Sys.getenv("AAD_APP_ID"),
        Sys.getenv("AAD_APP_SECRET")
      )
    }
  })
}

batch_get_pools <- function() {
  t <- batch_token_fn(BATCH_EP)
  pool_req <- httr::GET(
    Sys.getenv("BATCH_URL"),
    path = c("pools"),
    query = list("api-version" = "2022-01-01.15.0"),
    httr::add_headers(
      "Authorization" = paste("Bearer", AzureAuth::extract_jwt(t))
    )
  )

  df <- httr::content(pool_req) |>
    purrr::pluck("value") |>
    purrr::map_dfr(as.data.frame) |>
    tibble::as_tibble()

  if (nrow(df) == 0) {
    return(NULL)
  }

  df |>
    dplyr::select(
      .data$id,
      .data$state:.data$vmSize,
      .data$currentDedicatedNodes:.data$targetLowPriorityNodes
    )
}

batch_get_jobs <- function() {
  t <- batch_token_fn(BATCH_EP)
  jobs_req <- httr::GET(
    Sys.getenv("BATCH_URL"),
    path = c("jobs"),
    query = list("api-version" = "2022-01-01.15.0"),
    httr::add_headers(
      "Authorization" = paste("Bearer", AzureAuth::extract_jwt(t))
    )
  )

  df <- httr::content(jobs_req) |>
    purrr::pluck("value") |>
    purrr::map_dfr(as.data.frame) |>
    tibble::as_tibble()

  if (nrow(df) == 0) {
    return(NULL)
  }

  df |>
    dplyr::select(
      .data$id,
      .data$creationTime,
      .data$state,
      tidyselect::matches("executionInfo\\.(start|end)Time")
    ) |>
    dplyr::rename_with(
      stringr::str_remove,
      "^executionInfo\\.",
      .cols = tidyselect::matches("executionInfo\\.(start|end)Time")
    ) |>
    dplyr::mutate(
      dplyr::across(
        tidyselect::ends_with("Time"),
        lubridate::as_datetime
      )
    )
}

batch_get_tasks <- function(job_id) {
  t <- batch_token_fn(BATCH_EP)
  tasks_req <- httr::GET(
    Sys.getenv("BATCH_URL"),
    path = c("jobs", job_id, "tasks"),
    query = list("api-version" = "2022-01-01.15.0"),
    httr::add_headers(
      "Authorization" = paste("Bearer", AzureAuth::extract_jwt(t))
    )
  )

  df <- httr::content(tasks_req) |>
    purrr::pluck("value") |>
    purrr::map_dfr(as.data.frame) |>
    tibble::as_tibble()

  if (nrow(df) == 0) {
    return(NULL)
  }

  df |>
    dplyr::select(
      .data$id,
      .data$displayName,
      .data$state,
      .data$creationTime,
      tidyselect::matches("executionInfo\\.(start|end|result|exitCode)Time")
    ) |>
    dplyr::rename_with(
      stringr::str_remove,
      "^executionInfo\\.",
      .cols = tidyselect::starts_with("executionInfo\\.")
    ) |>
    dplyr::mutate(
      dplyr::across(
        tidyselect::ends_with("Time"),
        lubridate::as_datetime
      )
    ) |>
    dplyr::arrange(.data$id)
}

batch_add_job <- function(params) {
  sa_t <- batch_token_fn(STORAGE_EP)
  cont <- AzureStor::storage_container(
    glue::glue("{Sys.getenv('STORAGE_URL')}/queue"),
    token = AzureAuth::extract_jwt(sa_t)
  )

  # set create_datetime
  cdt <- Sys.time() |>
    lubridate::with_tz("UTC") |>
    format("%Y%m%d_%H%M%S")
  params[["create_datetime"]] <- cdt

  # create the name of the job and the filename
  job_name <- glue::glue("{params[['input_data']]}_{params[['name']]}_{cdt}")
  filename <- glue::glue("{job_name}.json")

  # upload the params to blob storage
  withr::local_file(filename)
  jsonlite::write_json(params, filename, auto_unbox = TRUE, pretty = TRUE)
  AzureStor::upload_blob(cont, filename)

  # create the job
  ba_t <- batch_token_fn(BATCH_EP)
  req <- httr::POST(
    Sys.getenv("BATCH_URL"),
    path = c("jobs"),
    body = list(
      id = job_name,
      poolInfo = list(poolId = "nhp-model"),
      onAllTasksComplete = "terminatejob",
      usesTaskDependencies = TRUE
    ),
    query = list(
      "api-version" = "2022-01-01.15.0"
    ),
    encode = "json",
    httr::add_headers(
      "Authorization" = paste("Bearer", AzureAuth::extract_jwt(ba_t)),
      "Content-Type" = "application/json;odata=minimalmetadata"
    )
  )

  # add tasks
  user_id <- list(
    "autoUser" = list(
      "scope" = "pool",
      "elevationLevel" = "admin"
    )
  )

  md <- "/mnt/batch/tasks/fsmounts"
  results_path <- glue::glue("{md}/results")
  temp_results_path <- glue::glue("{md}/batch/{uuid::UUIDgenerate()}")
  task_command <- function(run_start, runs_per_task) {
    glue::glue(
      .sep = " ",
      "/opt/nhp/bin/python",
      "{md}/app/run_model.py",
      "{md}/queue/{filename}",
      "--data-path={md}/data",
      "--results-path={results_path}",
      "--temp-results-path={temp_results_path}",
      "--save-type=cosmos",
      "--run-start={run_start}",
      "--model-runs={runs_per_task}"
    )
  }

  model_runs <- params[["model_runs"]]
  runs_per_task <- as.numeric(Sys.getenv("MODEL_RUNS_PER_TASK", 64))

  pad <- purrr::partial(
    stringr::str_pad,
    width = floor(log10(model_runs)) + 1,
    side = "left",
    pad = "0"
  )

  env_vars <- list(
    list(name = "COSMOS_ENDPOINT", value = Sys.getenv("COSMOS_ENDPOINT")),
    list(name = "COSMOS_KEY", value = Sys.getenv("COSMOS_KEY")),
    list(name = "COSMOS_DB", value = Sys.getenv("COSMOS_DB"))
  )

  task_fn <- function(run_start) {
    run_end <- run_start + runs_per_task - 1

    list(
      id = glue::glue("run_{pad(run_start)}-{pad(run_end)}"),
      displayName = glue::glue(
        "Model Run [{run_start} to {run_end}]"
      ),
      commandLine = task_command(run_start, runs_per_task),
      userIdentity = user_id
    )
  }

  tasks <- purrr::map(seq(1, model_runs, runs_per_task), task_fn)

  upload_to_cosmos <- list(
    id = "upload_to_cosmos",
    displayName = "Run Principal + Upload to Cosmos",
    commandLine = paste(task_command(-1, 2), "--run-postruns"),
    userIdentity = user_id,
    environmentSettings = env_vars,
    dependsOn = list(taskIds = purrr::map_chr(tasks, "id"))
  )

  clean_up_queue <- list(
    id = "clean_queue",
    displayName = "Clean up queue",
    commandLine = glue::glue("rm -rf {md}/queue/{filename}"),
    userIdentity = user_id,
    dependsOn = list(taskIds = c(purrr::map_chr(tasks, "id"), upload_to_cosmos$id))
  )

  all_tasks <- list(value = c(tasks, list(upload_to_cosmos, clean_up_queue)))

  httr::POST(
    Sys.getenv("BATCH_URL"),
    path = c("jobs", job_name, "addtaskcollection"),
    body = all_tasks,
    query = list(
      "api-version" = "2022-01-01.15.0"
    ),
    encode = "json",
    httr::add_headers(
      "Authorization" = paste("Bearer", AzureAuth::extract_jwt(ba_t)),
      "Content-Type" = "application/json;odata=minimalmetadata"
    )
  )

  return(job_name)
}