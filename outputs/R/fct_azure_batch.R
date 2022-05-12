token_fn <- function(resource) {
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

get_pools <- function() {
  t <- token_fn(BATCH_EP)
  pool_req <- httr::GET(
    Sys.getenv("BATCH_URL"),
    path = c("pools"),
    query = list("api-version" = "2022-01-01.15.0"),
    httr::add_headers(
      "Authorization" = paste("Bearer", AzureAuth::extract_jwt(t))
    )
  )

  httr::content(pool_req) |>
    purrr::pluck("value") |>
    purrr::map_dfr(as.data.frame) |>
    tibble::as_tibble() |>
    dplyr::select(
      .data$id,
      .data$state:.data$vmSize,
      .data$currentDedicatedNodes:.data$targetLowPriorityNodes
    )
}

get_jobs <- function() {
  t <- token_fn(BATCH_EP)
  jobs_req <- httr::GET(
    Sys.getenv("BATCH_URL"),
    path = c("jobs"),
    query = list("api-version" = "2022-01-01.15.0"),
    httr::add_headers(
      "Authorization" = paste("Bearer", AzureAuth::extract_jwt(t))
    )
  )

  httr::content(jobs_req) |>
    purrr::pluck("value") |>
    purrr::map_dfr(as.data.frame) |>
    tibble::as_tibble() |>
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

get_tasks <- function(job_id) {
  t <- token_fn(BATCH_EP)
  tasks_req <- httr::GET(
    Sys.getenv("BATCH_URL"),
    path = c("jobs", job_id, "tasks"),
    query = list("api-version" = "2022-01-01.15.0"),
    httr::add_headers(
      "Authorization" = paste("Bearer", AzureAuth::extract_jwt(t))
    )
  )

  httr::content(tasks_req) |>
    purrr::pluck("value") |>
    purrr::map_dfr(as.data.frame) |>
    tibble::as_tibble() |>
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

add_job <- function(params) {
  sa_t <- token_fn(STORAGE_EP)
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
  ba_t <- token_fn(BATCH_EP)
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
  run_results_path <- glue::glue("{md}/batch/{job_name}")
  task_command <- function(run_start, runs_per_task) {
    glue::glue(
      .sep = " ",
      "/opt/nhp/bin/python",
      "{md}/app/run_model.py",
      "{md}/queue/{filename}",
      "--data_path={md}/data",
      "--results_path={run_results_path}",
      "--run_start={run_start}",
      "--model_runs={runs_per_task}"
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

  task_fn <- function(run_start) {
    run_end <- run_start + runs_per_task - 1

    list(
      id = glue::glue("run_{pad(run_start)}-{pad(run_end)}"),
      displayName = glue::glue(
        "Model Run [{run_start} to {run_end}]"
      ),
      commandLine = task_command(run_start, runs_per_task),
      userIdentity = user_id,
      dependsOn = list(taskIds = principal_run$id)
    )
  }

  principal_run <- list(
    id = "principal_run",
    displayName = "Principal Model Run",
    commandLine = task_command(0, 1),
    userIdentity = user_id
  )

  tasks <- purrr::map(seq(1, model_runs, runs_per_task), task_fn)

  combine_command <- glue::glue(
    .sep = " ",
    "/opt/nhp/bin/python",
    "{md}/app/combine_results.py",
    "{run_results_path}",
    "{md}/results",
    params[["input_data"]],
    params[["name"]],
    cdt
  )

  combine_task <- list(
    id = "runs_combine",
    displayName = "Combine Results",
    commandLine = combine_command,
    userIdentity = user_id,
    dependsOn = list(taskIds = c(principal_run$id, purrr::map_chr(tasks, "id")))
  )

  remove_queue_task <- list(
    id = "runs_remove_queue",
    displayName = "Remove queue file",
    commandLine = glue::glue("rm {md}/queue/{filename}"),
    userIdentity = user_id,
    dependsOn = list(taskIds = combine_task$id)
  )

  all_tasks <- jsonlite::toJSON(
    list(
      value = c(
        list(principal_run, combine_task, remove_queue_task),
        tasks
      )
    ),
    pretty = TRUE,
    auto_unbox = TRUE
  ) |>
    stringr::str_replace_all("(?<=\"taskIds\": )\".*\"", "[\\0]")

  httr::POST(
    Sys.getenv("BATCH_URL"),
    path = c("jobs", job_name, "addtaskcollection"),
    body = all_tasks,
    query = list(
      "api-version" = "2022-01-01.15.0"
    ),
    encode = "raw",
    httr::add_headers(
      "Authorization" = paste("Bearer", AzureAuth::extract_jwt(ba_t)),
      "Content-Type" = "application/json;odata=minimalmetadata"
    )
  )

  return(job_name)
}
