#' submit_model_run
#'
#' @description A fct function
#'
#' @return The return value, if any, from executing the function.
#'
#' @noRd
submit_model_run <- function(params_json) {
  cat("submitting to batch\n")
  withr::local_dir("..")

  params_dir <- file.path(tempdir(), "queue") |>
    stringr::str_replace_all("\\\\", "/")
  dir.create(params_dir, "queue")
  params_file <- file.path(params_dir, "params.json")

  readr::write_lines(params_json, params_file)

  reticulate::use_condaenv("nhp")
  b <- reticulate::import("batch")

  b$prep_queue(256L, params_dir)
  cat("done\n")
}
