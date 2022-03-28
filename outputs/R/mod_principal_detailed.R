#' principal_detailed UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_principal_detailed_ui <- function(id) {
  activity_types <- list(
    "A&E" = "aae",
    "IP" = "ip",
    "OP" = "op"
  )

  ns <- NS(id)
  tagList(
    h1("Detailed activity estimates (principal projection)"),
    fluidRow(
      column(4, selectInput(ns("activity_type"), "Activity Type", activity_types)),
      column(4, selectInput(ns("pod"), "POD", NULL)),
      column(4, selectInput(ns("measure"), "Measure", NULL))
    ),
    gt::gt_output(ns("results"))
  )
}

#' principal_detailed Server Functions
#'
#' @noRd
mod_principal_detailed_server <- function(id, data) {
  moduleServer( id, function(input, output, session){
    ns <- session$ns

    age_groups <- tibble::tibble(age = 0:90) |>
      mutate(age_group = cut(
        .data$age,
        c(0, 5, 15, 35, 50, 65, 85, Inf),
        c("0 to 4",
          "5 to 14",
          "15 to 34",
          "35 to 49",
          "50 to 64",
          "65 to 84",
          "85+"
        ),
        right = FALSE
      ))

    data_fixed <- reactive({
      d <- data()

      aae <- d$aae |>
        dplyr::inner_join(age_groups, by = "age") |>
        dplyr::group_by(activity_type = "aae", .data$sex, .data$age_group, .data$type, .data$pod)

      ip <- d$ip |>
        dplyr::inner_join(age_groups, by = "age") |>
        dplyr::group_by(activity_type = "ip", .data$sex, .data$age_group, .data$type, .data$pod) |>
        dplyr::summarise(
          "Admissions" = dplyr::n(),
          "Mean Length of Stay" = mean(.data$speldur),
          "Total Bed Days" = sum(.data$beddays),
          .groups = "drop"
        ) |>
        tidyr::pivot_longer(c(Admissions:`Total Bed Days`), names_to = "measure")

      op <- d$op |>
        dplyr::inner_join(age_groups, by = "age") |>
        dplyr::group_by(activity_type = "op", .data$sex, .data$age_group, .data$type, .data$pod) |>
        dplyr::summarise(dplyr::across(dplyr::ends_with("attendances"), sum), .groups = "drop") |>
        tidyr::pivot_longer(dplyr::ends_with("attendances"), names_to = "measure")

      bind_rows(
        aae |>
          dplyr::filter(.data$aearrivalmode == "1") |>
          dplyr::summarise(measure = "Ambulance Arrivals", value = sum(.data$arrivals), .groups = "drop"),
        aae |>
          dplyr::filter(.data$aearrivalmode != "1") |>
          dplyr::summarise(measure = "Walk-in Arrivals", value = sum(.data$arrivals), .groups = "drop"),
        ip,
        op
      ) |>
        tidyr::pivot_wider(names_from = .data$type, values_from = .data$value) |>
        dplyr::rename(final = model) |>
        dplyr::mutate(change = final - baseline, change_pcnt = change / baseline)
    })

    selected_data <- reactive({
      at <- req(input$activity_type)
      p <- req(input$pod)
      m <- req(input$measure)

      data_fixed() |>
        dplyr::filter(
          .data$activity_type == at,
          .data$pod == p,
          .data$measure == m
        ) |>
        select(-activity_type, -pod, -measure)
    })

    output$results <- gt::render_gt({
      d <- selected_data()

      # handle some edge cases where a dropdown is changed and the next dropdowns aren't yet changed: we get 0 rows of
      # data which causes a bunch of warning messages
      req(nrow(d) > 0)

      d |>
        dplyr::mutate(
          dplyr::across(.data$sex, ~ifelse(.x == 1, "Male", "Female")),
          dplyr::across(.data$final, gt_bar, scales::comma_format(1), "#686f73", "#686f73"),
          dplyr::across(.data$change, gt_bar, scales::comma_format(1)),
          dplyr::across(.data$change_pcnt, gt_bar, scales::percent_format(1))
        ) |>
        gt::gt(groupname_col = "sex") |>
        gt::cols_label(
          age_group = "Age Group",
          baseline = "Baseline",
          final = "Final",
          change = "Change",
          change_pcnt = "Percent Change",
        ) |>
        gt::fmt_integer(c(baseline)) |>
        gt::cols_width(final ~ px(150), change ~ px(150), change_pcnt ~ px(150)) |>
        gt::cols_align(
          align = "left",
          columns = c("age_group", "final", "change", "change_pcnt")
        )
    })

    observeEvent(input$activity_type, {
      at <- req(input$activity_type)

      if (at == "aae") {
        p <- c(
          "Type 1" = "01",
          "Type 2" = "02",
          "Type 3" = "03",
          "Type 4" = "04",
          "Type Unknown" = "99"
        )
        m <- c("Ambulance Arrivals", "Walk-in Arrivals")
      } else if (at == "ip") {
        p <- c(
          "Elective" = "elective_admission",
          "Non-Elective" = "non-elective_admission",
          "Daycase" = "elective_daycase"
        )
        m <- c("Admissions", "Total Bed Days", "Mean Length of Stay")
      } else if (at == "op") {
        p <- c(
          "OP First" = "op_first",
          "OP Follow-up" = "op_follow-up",
          "OP Procedures" = "op_procedure"
        )
        m <- c("Attendances" = "attendances", "Tele-Attendances" = "tele_attendances")
      }
      updateSelectInput(session, "pod", choices = p)
      updateSelectInput(session, "measure", choices = m)
    })
  })
}
