.data <- NULL # lint helper

extract_reference_data <- function(path) {
  con <- DBI::dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR_REFERENCE")
  )
  withr::defer(DBI::dbDisconnect(con))

  dplyr::tbl(con, dbplyr::in_schema("dbo", "DIM_tbDiagnosis")) |>
    dplyr::filter(.data$DiagnosisId > 0) |>
    dplyr::select(
      icd10 = "DiagnosisCode",
      diagnosis = "DiagnosisDescription",
      chapter = "ChapterCode",
      chapter_desc = "ChapterDescription",
      subchapter = "SubChapterCode",
      subchapter_desc = "SubChapterDescription"
    ) |>
    dplyr::collect() |>
    readr::write_csv(file.path(path, "icd10.csv"))

  dplyr::tbl(con, dbplyr::in_schema("dbo", "DIM_tbProcedure")) |>
    dplyr::filter(.data$ProcedureId > 0) |>
    dplyr::select(
      opcs4 = "ProcedureCode",
      procedure = "ProcedureDescription"
    ) |>
    dplyr::collect() |>
    readr::write_csv(file.path(path, "opcs4.csv"))

  Sys.time()
}
