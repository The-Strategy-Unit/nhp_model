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
  ~pod,                            ~pod_display,
  "aae_type-01",                   "Type 1",
  "aae_type-02",                   "Type 2",
  "aae_type-03",                   "Type 3",
  "aae_type-04",                   "Type 4",
  "aae_type-99",                   "Type Unknown",
  "ip_elective_admission",         "Elective Admission",
  "ip_elective_daycase",           "Daycase Admission",
  "ip_non-elective_admission",     "Non-Elective Admission",
  "ip_non-elective_birth-episode", "Birth Episode",
  "op_first",                      "First Attendance",
  "op_follow-up",                  "Follow up Attendance",
  "op_procedure",                  "Procedure"
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