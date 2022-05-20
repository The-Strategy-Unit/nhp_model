#' @importFrom zeallot %<-%
#' @importFrom rlang .data
NULL

# converts a two column tibble into a named list suitable for shiny selectInput choices
set_names <- function(.x) {
  purrr::set_names(.x[[1]], .x[[2]])
}

# source: https://github.com/r-lib/tidyselect/issues/201#issuecomment-650547846
utils::globalVariables("where")

BATCH_EP <- "https://batch.core.windows.net/"
STORAGE_EP <- "https://storage.azure.com/"

model_run_type <- function(mr) {
  case_when(
    mr == -1 ~ "baseline",
    mr == 0 ~ "principal",
    mr > 0 ~ "model"
  )
}