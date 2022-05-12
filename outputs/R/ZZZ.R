#' @importFrom zeallot %<-%
#' @importFrom rlang .data
NULL

# source: https://github.com/r-lib/tidyselect/issues/201#issuecomment-650547846
utils::globalVariables("where")

BATCH_EP <- "https://batch.core.windows.net/"
STORAGE_EP <- "https://storage.azure.com/"