library(tidyverse)
library(DBI)
library(dbplyr)

extract_reference_data <- function(path) {
  con <- DBI::dbConnect(
    odbc::odbc(),
    .connection_string = Sys.getenv("CONSTR_REFERENCE")
  )
  withr::defer(DBI::dbDisconnect(con))

  tbl(con, in_schema("dbo", "DIM_tbDiagnosis")) |>
    filter(DiagnosisId > 0) |>
    select(
      icd10 = DiagnosisCode,
      diagnosis = DiagnosisDescription,
      chapter = ChapterCode,
      chapter_desc = ChapterDescription,
      subchapter = SubChapterCode,
      subchapter_desc = SubChapterDescription
    ) |>
    collect() |>
    write_csv(file.path(path, "icd10.csv"))

  opcs4 <- tbl(con, in_schema("dbo", "DIM_tbProcedure")) |>
    filter(ProcedureId > 0) |>
    select(
      opcs4 = ProcedureCode,
      procedure = ProcedureDescription
    ) |>
    collect() |>
    write_csv(here::here("data", "opcs4.csv"))
}
