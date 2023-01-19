upload_file_to_azure <- function(file, data_version) {
  ep <- AzureStor::adls_endpoint(
    endpoint = Sys.getenv("TARGETS_AZURE_SA_EP"),
    key = Sys.getenv("TARGETS_AZURE_SA_key")
  )
  fs <- AzureStor::adls_filesystem(ep, "data")

  AzureStor::upload_adls_file(fs, file, stringr::str_replace(file, "^data/", data_version))

  tibble::tibble(file = file, uploaded = Sys.time())
}
