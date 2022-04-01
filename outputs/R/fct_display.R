#' display
#'
#' @description A fct function
#'
#' @return The return value, if any, from executing the function.
#'
#' @noRd
dataset_display <- tibble::tribble(
  ~dataset, ~dataset_display,
  "aae",    "A&E",
  "ip",     "Inpatients",
  "op",     "Outpatients"
)

pod_display <- tibble::tribble(
  ~pod,                         ~pod_display,
  "type-01",                    "Type 1",
  "type-02",                    "Type 2",
  "type-03",                    "Type 3",
  "type-04",                    "Type 4",
  "type-99",                    "Type Unknown",
  "elective_admission",         "Elective Admission",
  "elective_daycase",           "Daycase Admission",
  "non-elective_admission",     "Non-Elective Admission",
  "non-elective_birth-episode", "Birth Episode",
  "op_first",                   "First Attendance",
  "op_follow-up",               "Follow up Attendance",
  "op_procedure",               "Procedure"
)

measure_display <- tibble::tribble(
  ~measure,           ~measure_display,
  "ambulance",        "Ambulance Arrivals",
  "walk-in",          "Walk-in Arrivals",
  "admissions",       "Admissions",
  "beddays",          "Bed Days",
  "attendances",      "Face-to-Face Appointment",
  "tele_attendances", "Tele Appointment"
)
